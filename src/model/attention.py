# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
import math
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from einops import rearrange, repeat

from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import Attention as CrossAttention, FeedForward, AdaLayerNorm
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
@dataclass
class Transformer3DModelOutput(BaseOutput):
    """Output of the Transformer3DModel.
    
    Args:
        sample (`torch.FloatTensor`): Output tensor from the transformer model.
    """
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class MutitaskAttention(nn.Module):
    """Multi-task attention module that processes task-specific features.
    
    This module extends standard attention mechanisms to handle multiple tasks
    by creating task-specific key and value projections for each output type.
    
    Args:
        attn: Base attention module to extend with multi-task capabilities.
        output_types (List[str]): List of output types/tasks to support.
        init_as_pretrained (bool, optional): Whether to initialize task-specific
            projections with weights from the base attention module. Defaults to True.
    """
    def __init__(self, attn, output_types, init_as_pretrained=True):
        super().__init__()
        self.attn = attn
        
        # Add task-specific projections
        self.task_to_k = nn.ModuleDict()
        self.task_to_v = nn.ModuleDict()

        for output_type in output_types:
            self.task_to_k[output_type] = nn.Linear(attn.to_k.in_features, attn.to_k.out_features, bias=False)
            self.task_to_v[output_type] = nn.Linear(attn.to_v.in_features, attn.to_v.out_features, bias=False)

            if init_as_pretrained:
                self.task_to_k[output_type].weight = nn.Parameter(attn.to_k.weight.clone())
                self.task_to_v[output_type].weight = nn.Parameter(attn.to_v.weight.clone())

        # Initialize query projection and output layers
        self.task_to_q = nn.Linear(attn.to_q.in_features, attn.to_q.out_features, bias=False)
        if init_as_pretrained:
            self.task_to_q.weight = nn.Parameter(attn.to_q.weight.clone())

        self.to_out_task = nn.Linear(attn.to_out[0].in_features, attn.to_out[0].out_features, bias=True)
        nn.init.zeros_(self.to_out_task.weight)
        nn.init.zeros_(self.to_out_task.bias)

        self.norm_main = nn.LayerNorm(attn.to_out[0].out_features)

    def forward(self, norm_hidden_states, task_feat):
        """Forward pass for multi-task attention.
        
        Args:
            norm_hidden_states (torch.Tensor): Normalized hidden states from the model.
            task_feat (Dict[str, torch.Tensor]): Dictionary mapping task names to their feature tensors.
            
        Returns:
            torch.Tensor: Processed attention output for the tasks.
        """
        # Task-specific processing
        task_keys = []
        task_values = []
        for output_type in task_feat.keys():
            task_hidden = task_feat[output_type]
            task_keys.append(self.task_to_k[output_type](task_hidden))
            task_values.append(self.task_to_v[output_type](task_hidden))

        task_key = torch.stack(task_keys, dim=1)
        task_value = torch.stack(task_values, dim=1)
        batch_size = task_key.size(0)
        
        # Rearrange dimensions for attention
        task_key = rearrange(task_key, "b f d c -> (b d) f c")
        task_value = rearrange(task_value, "b f d c -> (b d) f c")

        # Process task queries
        task_query = self.task_to_q(norm_hidden_states)
        task_query = rearrange(task_query.unsqueeze(1), "b f d c -> (b d) f c")

        # Apply attention to task features
        task_attn = self.attn.apply_attention(task_query, task_key, task_value)
        task_attn = rearrange(task_attn, "(b d) f c -> b f d c", b=batch_size).squeeze(1)

        # Combine with main attention output
        return self.to_out_task(task_attn)


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
                for d in range(num_layers)
            ]
        )

        assert num_layers == 1, "Only one transformer block is supported for now"

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True, task_feat=None, output_type=None):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        # encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)
        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            assert len(self.transformer_blocks) == 1, "Only one transformer block is supported for now"
            hidden_states, ret_task_feats = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
                task_feat=task_feat,
                output_type=output_type
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output), ret_task_feats


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.attention_bias = attention_bias
        self.upcast_attention = upcast_attention
        self.activation_fn = activation_fn

        # SC-Attn
        self.attn1 = SparseCausalAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
    
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        self.attn_temp = None
      
    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, attention_op=None):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e

            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            self.attn1.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
                self.attn2.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)
            if self.attn_temp is not None:
                self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
                self.attn_temp.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)

    def forward(self, hidden_states, encoder_hidden_states=None,
                timestep=None, attention_mask=None,
                output_type=None,
                video_length=None, task_feat=None):
        # SparseCausal-Attention
        ret_task_feats = None

        if not hasattr(self, 'return_feature'):
            self.return_feature = None
        
        if self.return_feature == "beforeSelfAttn":
            ret_task_feats = hidden_states
            # print("beforeSelfAttn", ret_task_feats.shape)

        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )
        if self.only_cross_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
            )
        else:
            self_attn_hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length,
                                                           task_feat=task_feat, output_type=output_type)
            hidden_states = self_attn_hidden_states + hidden_states
            if self.return_feature == "afterSelfAttn_residual":
                ret_task_feats = self_attn_hidden_states
            elif self.return_feature == "afterSelfAttn_main":
                ret_task_feats = hidden_states
            


        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            x_attn_hidden_states = self.attn2(
                norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
            )
            
            hidden_states = x_attn_hidden_states + hidden_states 
            if self.return_feature == "afterXAttn_residual":
                ret_task_feats = x_attn_hidden_states
            elif self.return_feature == "afterXAttn_main":
                ret_task_feats = hidden_states


        # Feed-forward
        ff_hidden_states = self.ff(self.norm3(hidden_states)) 
        hidden_states = ff_hidden_states + hidden_states
        
        if self.return_feature == "afterFF_residual":
            ret_task_feats = ff_hidden_states
        elif self.return_feature == "afterFF_main":
            ret_task_feats = hidden_states

        return hidden_states, ret_task_feats


class SparseCausalAttention(CrossAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_mask_ratio = 0.1
        self.temperature = 100
        self.n_attns = 2
        
    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        query = query.contiguous().to(torch.float16)
        key = key.contiguous().to(torch.float16)
        value = value.contiguous().to(torch.float16)
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = self.batch_to_head_dim(hidden_states)
        return hidden_states


    def set_apply_task_attn_to_layers(self, layer_name="all"):
        assert layer_name in ["all", "dec"]
        
        if layer_name == "all":
            self.apply_task_attn_to_layers = set(range(16))
        elif layer_name == "dec":
            self.apply_task_attn_to_layers = set([7, 8, 9, 10, 11, 12, 13, 14, 15])
        print("set_apply_task_attn_to_layers", layer_name, self.apply_task_attn_to_layers)
        
        
    def apply_attention(self, query, key, value, attention_mask):
        _, sequence_length, dim = query.shape
        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)
        assert self._use_memory_efficient_attention_xformers == True, "memory efficient attention is not set to True"
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
        return hidden_states


    def forward(self, hidden_states, output_type, encoder_hidden_states=None,
                attention_mask=None, video_length=None, task_feat=None):
        
        n_attns = self.n_attns
        batch_size, sequence_length, _ = hidden_states.shape
    

        encoder_hidden_states = encoder_hidden_states


        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)


        query = self.to_q(hidden_states)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError


        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)




        hidden_states = self.apply_attention(query, key, value, attention_mask)
        

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        
        if task_feat is not None:
            assert output_type is not None, "output_type is required"

            
            task_feat_idx = self.current_layer_idx
            
            
            if task_feat_idx in self.apply_task_attn_to_layers:
                _task_feat = task_feat[task_feat_idx]
                task_keys = []
                task_values = []
                task_output_types = list(_task_feat.keys())
                for task_output_type in task_output_types:
                    task_hidden_states = _task_feat[task_output_type] # B, n, d
                    if isinstance(self.task_to_v, nn.ModuleDict):
                        _task_to_v = self.task_to_v[task_output_type]
                        _task_norm_v = self.task_norm_v[task_output_type]
                        _task_to_k = self.task_to_k[task_output_type]
                        _task_norm_k = self.task_norm_k[task_output_type]
                    else:
                        _task_to_v = self.task_to_v
                        _task_norm_v = self.task_norm_v
                        _task_to_k = self.task_to_k
                        _task_norm_k = self.task_norm_k
                    
                    
                    if hasattr(self, "next_layer_idx") and self.next_layer_idx != -1:
                        _task_feat1 = task_feat[self.next_layer_idx]
                        task_hidden_states1 = _task_feat1[task_output_type] # B, n, d
                        task_hidden_states = torch.cat([task_hidden_states, task_hidden_states1], dim=2)
                        
                    task_value = _task_to_v(_task_norm_v(task_hidden_states))
                    task_key = _task_to_k(_task_norm_k(task_hidden_states))
                    task_keys.append(task_key)
                    task_values.append(task_value)
                    
                
                task_key = torch.stack(task_keys, dim=1)
                task_value = torch.stack(task_values, dim=1)
                task_key = rearrange(task_key, "b f d c -> (b d) f c")
                task_value = rearrange(task_value, "b f d c -> (b d) f c")
                

                if isinstance(self.task_to_q, nn.ModuleDict):
                    _task_to_q = self.task_to_q[output_type]
                    _task_norm_q = self.task_norm_q[output_type]
                else:
                    _task_to_q = self.task_to_q
                    _task_norm_q = self.task_norm_q
                task_query = _task_to_q(_task_norm_q(hidden_states))
          
                task_query = task_query.unsqueeze(1)
                task_query = rearrange(task_query, "b f d c -> (b d) f c")

                task_query = self.head_to_batch_dim_with_head_size(task_query, n_attns)
                task_key = self.head_to_batch_dim_with_head_size(task_key, n_attns)
                task_value = self.head_to_batch_dim_with_head_size(task_value, n_attns)
            
            
                mask_task_idx = None
                with torch.no_grad():
                    # Compute attention scores from task_query and task_key
                    random_number = random.random()
                    if self.training and random_number < self.attn_mask_ratio:
                        if task_key.shape[1] > 8096:
                            indices = torch.randperm(task_key.shape[1], device=task_key.device)[:8096]
                            _task_key = task_key[:, indices, :]
                            _task_query = task_query[:, indices, :]
                        else:
                            _task_key = task_key
                            _task_query = task_query
                        scores = torch.matmul(_task_query, _task_key.transpose(-2, -1)) / math.sqrt(_task_query.size(-1))
                        scores = torch.softmax(scores, dim=-1).squeeze(1)
                        mean_scores = scores.mean(dim=0)
                        
                        if self.attn_mask_type == "attn_prob":
                            mask_task_idx = torch.multinomial(mean_scores, num_samples=1)
                        elif self.attn_mask_type == "random":
                            mask_task_idx = torch.randint(0, len(mean_scores), (1,), device=task_key.device)
                        elif self.attn_mask_type == "highest":
                            mask_task_idx = torch.argmax(mean_scores).unsqueeze(0)  # Convert single index to tensor
                        elif self.attn_mask_type == "attn_prob_random_k":
                            num_samples = torch.randint(1, len(mean_scores), (1,), device=task_key.device)
                            mask_task_idx = torch.multinomial(mean_scores, num_samples=num_samples.item())
                        else:
                            raise ValueError(f"Invalid attn_mask_type: {self.attn_mask_type}")
                    
                        
                # task_hidden_states = self.apply_attention(task_query, task_key, task_value, attention_mask)

                slice_size = 2048
                num_chunks = task_query.shape[0] // slice_size + 1
                chunked_task_query = torch.chunk(task_query, num_chunks, dim=0)
                chunked_task_key = torch.chunk(task_key, num_chunks, dim=0)
                chunked_task_value = torch.chunk(task_value, num_chunks, dim=0)
                
                temp_attn_slices = []
                for chunk_query, chunk_key, chunk_value in zip(chunked_task_query, chunked_task_key, chunked_task_value):

                    attn_bias = None
                    # Randomly mask out one task for each sample
                    # B, 1, n_tasks
                    # Create attention bias with proper padding to ensure stride is a multiple of 8
                    # First determine the padded size that's a multiple of 8
                    if mask_task_idx is not None:
                        pad_to_length = ((chunk_key.shape[1] + 7) // 8) * 8

                        # Create properly padded tensor
                        attn_bias = torch.zeros(
                            chunk_query.shape[0], chunk_query.shape[1], pad_to_length,
                            device=chunk_query.device, dtype=chunk_query.dtype
                        )

                        batch_index = torch.arange(chunk_query.shape[0], device=chunk_query.device)

                        
                        for idx in mask_task_idx:
                            attn_bias[batch_index, :, idx] = float('-inf')
                        
                        # Use only the needed portion of the padded tensor
                        attn_bias = attn_bias[:, :, :chunk_key.shape[1]]


                    # Apply attention with the properly formatted bias
                    temp_attn_slice = xformers.ops.memory_efficient_attention(
                        chunk_query, 
                        chunk_key, 
                        chunk_value,
                        attn_bias=attn_bias
                    )

                    temp_attn_slices.append(temp_attn_slice.to(chunk_query.dtype))
                temp_attn_slices = torch.cat(temp_attn_slices, dim=0)
                temp_attn_slices = self.batch_to_head_dim_with_head_size(temp_attn_slices, n_attns)
                task_hidden_states = rearrange(temp_attn_slices, "(b d) f c -> b f d c", b=batch_size).squeeze(1)
                task_hidden_states = self.to_out_task(task_hidden_states)
                
                hidden_states = hidden_states + task_hidden_states

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
    

    def batch_to_head_dim_with_head_size(self, tensor: torch.Tensor, head_size: int) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim_with_head_size(self, tensor: torch.Tensor, head_size: int) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        b, s, d = tensor.shape
        tensor = tensor.reshape(b, s, head_size, d//head_size)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(b*head_size, s, d//head_size)
        return tensor

    @staticmethod
    def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
        num_chunks = hidden_states.shape[chunk_dim] // chunk_size + 1
        chunked_hidden_states = torch.chunk(hidden_states, num_chunks, dim=chunk_dim)
        # ===
        chunked_ff_output = []
        for chunk_hidden_states in chunked_hidden_states:
            chunked_ff_output.append(ff(chunk_hidden_states))
        chunked_ff_output = torch.cat(chunked_ff_output, dim=chunk_dim)
        return chunked_ff_output



class MLP(nn.Module):
    """Simple Multi-Layer Perceptron (MLP) module.
    
    A two-layer MLP with configurable hidden dimensions and activation function.
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        hidden_features (int, optional): Number of hidden features. Defaults to out_features.
        act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        init_as_zeros (bool, optional): Whether to initialize weights and biases to zeros. Defaults to False.
    """
    def __init__(self, in_features, out_features, hidden_features=None, act_layer=nn.GELU, init_as_zeros=False):
        super().__init__()
        # If hidden_features is not provided, default to out_features
        if hidden_features is None:
            hidden_features = out_features

        # First linear layer maps input features to hidden features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Activation layer (default is GELU)
        self.act = act_layer()
        # Second linear layer maps hidden features to output features
        self.fc2 = nn.Linear(hidden_features, out_features)

        if init_as_zeros:
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MLPv2(nn.Module):
    """
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        hidden_features (int, optional): Number of hidden features. Defaults to out_features.
        num_hidden_layers (int, optional): Number of hidden layers. Defaults to 1.
        act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        init_as_zeros (bool, optional): Whether to initialize weights and biases to zeros. Defaults to False.
    """
    def __init__(self, in_features, out_features, 
                 hidden_features=None, num_hidden_layers=1, act_layer=nn.GELU, init_as_zeros=False):
        super().__init__()
        # If hidden_features is not provided, default to out_features
        if hidden_features is None:
            hidden_features = out_features
        
        self.act = act_layer()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(self.act)
        
        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(self.act)
            
        # Output layer
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*layers)

        if init_as_zeros:
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Forward pass through the MLPv2 network.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        return self.net(x)