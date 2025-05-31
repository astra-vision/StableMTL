from src.model.unet import InflatedConv3d
from torch.nn.parameter import Parameter
from src.model.attention import SparseCausalAttention, MutitaskAttention, MLP, MLPv2
import torch.nn as nn
import torch
from diffusers.models.attention import FeedForward, Attention as CrossAttention, BasicTransformerBlock
from src.model.unet import UNet3DConditionModel
import os


def _replace_unet_conv_in(model, repeat=3):
    _weight = model.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = model.conv_in.bias.clone()  # [320]
    _weight = _weight.repeat((1, repeat, 1, 1))  # Keep selected channel(s)
    _weight *= 1 / repeat

    _n_convin_out_channel = model.conv_in.out_channels
    _new_conv_in = InflatedConv3d(
        4 * repeat, _n_convin_out_channel, kernel_size=3, padding=(1, 1))
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    model.conv_in = _new_conv_in
    print("Unet conv_in layer is replaced")
    # replace config
    model.config["in_channels"] = 4 * repeat
    print("Unet config is updated with repeat:", repeat)
    return


def config_unet_child(model, return_feature="selfAttn_residual"):
    assert return_feature in [
        "beforeSelfAttn",
        "afterSelfAttn_main", "afterSelfAttn_residual",
        "afterXAttn_main", "afterXAttn_residual",
        "afterFF_main", "afterFF_residual"
    ], f"Invalid return feature: {return_feature}"
    print("Configuring unet_child")
    print(f"Return feature: {return_feature}")
    block_idx = 0
    blocks = [*model.down_blocks, model.mid_block, *model.up_blocks]
    for block in blocks:
        if hasattr(block, 'attentions'):  # Check if the block has attention layers
            for attn in block.attentions:
                for transformer_block in attn.transformer_blocks:
                    transformer_block.return_feature = return_feature
                block_idx += 1


def _dupplicate_key_val_mlp_in_sparse_causal_attn(
    model,
    output_types,
    attn_mask_ratio=0.4,
    n_attns=4,
    attn_mask_type="attn_prob",
    apply_task_attn_to_layers="all"
):
    """Set up the sparse causal attention mechanism in the UNet model.
    
    Args:
        model: The UNet model to modify
        output_types: Types of outputs to support
        attn_mask_ratio: Ratio for attention masking
        n_attns: Number of attention heads
        attn_mask_type: Type of attention mask to use
        apply_task_attn_to_layers: Which layers to apply task attention to
    """
    layer_dims = [
        320, 320,  # 1728, 1728
        640, 640,  # 432, 432
        1280, 1280,  # 108, 108
        1280,  # 30
        1280, 1280, 1280,  # 108, 108, 108
        640, 640, 640,  # 432, 432, 432
        320, 320, 320  # 1728, 1728, 1728
    ]
    attn_to_idx = [
        0, 1,
        2, 3,
        4, 5,
        6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15
    ]
    attn_idx = 0
    block_idx = 0
    blocks = [*model.down_blocks, model.mid_block, *model.up_blocks]
    for block in blocks:
        if hasattr(block, 'attentions'):
            for attn in block.attentions:
                for transformer_block in attn.transformer_blocks:
                    sparse_attn = transformer_block.attn1
                    if isinstance(sparse_attn, SparseCausalAttention):
                        sparse_attn.n_attns = n_attns
                        sparse_attn.set_apply_task_attn_to_layers(
                            apply_task_attn_to_layers)
                        print(f"Found SparseCausalAttention in block {block_idx}, attention layer {attn_idx}")
                        sparse_attn.attn_mask_ratio = attn_mask_ratio
                        sparse_attn.attn_mask_type = attn_mask_type
                        print(f"SparseCausalAttention n_attns: {sparse_attn.n_attns}, attn_mask_type: {sparse_attn.attn_mask_type}, attn_mask_ratio: {sparse_attn.attn_mask_ratio}")

                        sparse_attn.task_to_k = nn.ModuleDict()
                        sparse_attn.task_to_v = nn.ModuleDict()
                        sparse_attn.task_to_q = nn.ModuleDict()
                        sparse_attn.task_norm_k = nn.ModuleDict()
                        sparse_attn.task_norm_v = nn.ModuleDict()
                        sparse_attn.task_norm_q = nn.ModuleDict()

                        
                        print(f"Setting up separate QKV for layer {attn_idx}")
                        current_layer_dim = layer_dims[attn_idx]
                        sparse_attn.current_layer_idx = attn_to_idx[attn_idx]
                        for output_type in output_types:
                            sparse_attn.task_to_k[output_type] = MLP(
                                current_layer_dim,
                                current_layer_dim,
                                hidden_features=current_layer_dim // 2,
                                init_as_zeros=False
                            )
                            sparse_attn.task_to_v[output_type] = MLP(
                                current_layer_dim,
                                current_layer_dim,
                                hidden_features=current_layer_dim // 2,
                                init_as_zeros=False
                            )
                            sparse_attn.task_to_q[output_type] = MLPv2(
                                current_layer_dim,
                                current_layer_dim,
                                num_hidden_layers=2,
                                hidden_features=640,
                                init_as_zeros=False
                            )
                            sparse_attn.task_norm_k[output_type] = nn.LayerNorm(
                                current_layer_dim)
                            sparse_attn.task_norm_v[output_type] = nn.LayerNorm(
                                current_layer_dim)
                            sparse_attn.task_norm_q[output_type] = nn.LayerNorm(
                                current_layer_dim)

                        sparse_attn.to_out_task = nn.Linear(
                            sparse_attn.to_out[0].in_features,
                            sparse_attn.to_out[0].out_features,
                            bias=True
                        )
                        nn.init.zeros_(sparse_attn.to_out_task.weight.data)
                        nn.init.zeros_(sparse_attn.to_out_task.bias.data)

                        attn_idx += 1
            block_idx += 1


def setup_unet(model, pretrained_model_path, output_types, cfg):
    """Setup UNet model with configuration parameters from cfg.
    
    Args:
        model: The model to setup UNet for
        pretrained_model_path: Path to pretrained model
        output_types: Types of outputs
        cfg: Configuration object containing model parameters
    """
    # Extract all parameters from config
    unet_weight_path = None
    if hasattr(cfg.trainer, 'unet_weight_path') and cfg.trainer.unet_weight_path is not None:
        unet_weight_path = os.path.join(cfg.trainer.unet_weight_path)

    apply_task_attn_to_layers = None
    if hasattr(cfg.trainer, 'apply_task_attn_to_layers') and cfg.trainer.apply_task_attn_to_layers is not None:
        apply_task_attn_to_layers = cfg.trainer.apply_task_attn_to_layers

    attn_mask_ratio = 0.0
    if hasattr(cfg.trainer, 'attn_mask_ratio') and cfg.trainer.attn_mask_ratio is not None:
        attn_mask_ratio = cfg.trainer.attn_mask_ratio

    attn_mask_type = "attn_prob"
    if hasattr(cfg.trainer, 'attn_mask_type') and cfg.trainer.attn_mask_type is not None:
        attn_mask_type = cfg.trainer.attn_mask_type

    main_stream_from_scratch = False
    if hasattr(cfg.trainer, 'main_stream_from_scratch') and cfg.trainer.main_stream_from_scratch is not None:
        main_stream_from_scratch = cfg.trainer.main_stream_from_scratch
        
    return_feature = "afterSelfAttn_residual"
    if hasattr(cfg.trainer, 'return_feature') and cfg.trainer.return_feature is not None:
        return_feature = cfg.trainer.return_feature
        
    n_attns = 4
    if hasattr(cfg.trainer, 'n_attns') and cfg.trainer.n_attns is not None:
        n_attns = cfg.trainer.n_attns

    repeat_input = 3
    if hasattr(cfg.pipeline.kwargs, 'encode_rgb_model') and cfg.pipeline.kwargs.encode_rgb_model is not None:
        if cfg.pipeline.kwargs.encode_rgb_model == "avg":
            repeat_input = 2


    model.unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path, subfolder="unet")
    model.unet.set_use_memory_efficient_attention_xformers(True)
    _replace_unet_conv_in(model.unet, repeat=repeat_input)

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(True)

    if unet_weight_path is not None:
        print("Loading unet_child from: ", unet_weight_path)

        model.unet_child = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path, subfolder="unet")
        model.unet_child.set_use_memory_efficient_attention_xformers(True)
        _replace_unet_conv_in(model.unet_child, repeat=repeat_input)
        config_unet_child(model.unet_child, return_feature=return_feature)

        if not main_stream_from_scratch:
            print("Loading unet from: ", unet_weight_path)
            model.unet.load_state_dict(torch.load(unet_weight_path))
        else:
            print("Main stream from stable diffusion 2")

        model.unet_child.load_state_dict(torch.load(unet_weight_path))
        model.unet_child.eval()

        model.unet_child.requires_grad_(False)

        _dupplicate_key_val_mlp_in_sparse_causal_attn(model.unet,
                                                      output_types=output_types,
                                                      n_attns=n_attns,
                                                      apply_task_attn_to_layers=apply_task_attn_to_layers,
                                                      attn_mask_ratio=attn_mask_ratio,
                                                      attn_mask_type=attn_mask_type)

    else:
        model.unet_child = None
