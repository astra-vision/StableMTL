import logging
import random
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from src.dataset.semantic import VKitti2Encoder
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)


class StableMTLDepthOutput(BaseOutput):
    """
    Output class for StableMTL monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
    """

    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]


class StableMTLNormalOutput(BaseOutput):
    """
    Output class for StableMTL monocular normal prediction pipeline.

    Args:
        normal_np (`np.ndarray`):
            Predicted normal map, with normal values in the range of [-1, 1].
        normal_colored (`PIL.Image.Image`):
            Colorized normal map, with the shape of [3, H, W] and values in [0, 1].
    """

    normal_np: np.ndarray
    normal_colored: Union[None, Image.Image]

class StableMTLSemsegOutput(BaseOutput):
    """
    Output class for StableMTL semantic segmentation prediction pipeline.
    
    Args:
        semantic_class_id (`np.ndarray`):
            Predicted semantic segmentation map with class IDs.
    """
    semantic_class_id: np.ndarray

class StableMTLOpticalFlowOutput(BaseOutput):
    """
    Output class for StableMTL optical flow prediction pipeline.
    
    Args:
        optical_flow_np (`np.ndarray`):
            Predicted optical flow map with 2-channel flow vectors.
    """
    optical_flow_np: np.ndarray

class StableMTLSceneFlowOutput(BaseOutput):
    """
    Output class for StableMTL scene flow prediction pipeline.
    
    Args:
        scene_flow_np (`np.ndarray`):
            Predicted scene flow map with 3D flow vectors.
    """
    scene_flow_np: np.ndarray

class StableMTLAlbedoOutput(BaseOutput):
    """
    Output class for StableMTL albedo prediction pipeline.
    
    Args:
        albedo_np (`np.ndarray`):
            Predicted albedo map representing surface reflectance properties.
    """
    albedo_np: np.ndarray

class StableMTLShadingOutput(BaseOutput):
    """
    Output class for StableMTL shading prediction pipeline.
    
    Args:
        shading_np (`np.ndarray`):
            Predicted shading map representing illumination effects.
    """
    shading_np: np.ndarray


class StableMTLPipeline(DiffusionPipeline):
    """
    Multi-task learning diffusion pipeline for vision tasks, including depth, normals, semantic segmentation, optical flow, scene flow, albedo, and shading prediction.

    This pipeline leverages a conditional U-Net, VAE, scheduler, and CLIP-based text encoder/tokenizer to generate predictions for a range of pixel-level tasks from single or paired input images. The model supports scale- and shift-invariant inference and can be configured for deterministic or random noise injection and different RGB encoding modes.

    Args:
        unet (`UNet2DConditionModel`): Conditional U-Net module for denoising task latents.
        vae (`AutoencoderKL`): Variational Autoencoder for encoding/decoding image and task representations.
        scheduler (`Union[DDIMScheduler, LCMScheduler]`): Diffusion scheduler for controlling the denoising process.
        text_encoder (`CLIPTextModel`): CLIP text encoder for conditional embeddings.
        tokenizer (`CLIPTokenizer`): Tokenizer for prompt encoding.
        scale_invariant (`bool`, optional): If True, output is invariant to scale. Default is True.
        shift_invariant (`bool`, optional): If True, output is invariant to shift. Default is True.
        default_denoising_steps (`int`, optional): Default number of denoising steps if not specified during inference.
        default_processing_resolution (`int`, optional): Default input resolution for preprocessing.
        input_noise (`str`, optional): Noise injection mode, 'random' or 'deterministic'. Default is 'deterministic'.
        encode_rgb_model (`str`, optional): RGB encoding strategy, e.g., 'duplicate', 'zero', or 'avg'. Default is 'duplicate'.

    Use this pipeline to perform multi-task vision inference or training, supporting both single-frame and paired-frame tasks, with configurable task conditioning and noise modeling.
    """

    rgb_latent_scale_factor = 0.18215
    latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
        input_noise: Optional[str] = "deterministic",
        encode_rgb_model: Optional[str] = "duplicate",
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None
        self.input_noise = input_noise
        self.semantic_encoder = VKitti2Encoder(n_classes=8)
        self.encode_rgb_model = encode_rgb_model

    @torch.no_grad()
    def __call__(
        self,
        input_image: torch.Tensor,
        exclude_mainstream_output_type: bool,
        next_input_image: Optional[torch.Tensor] = None,
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        output_type: str = "depth",
        task_output_types: List[str] = [],
    ) -> BaseOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`torch.Tensor`):
                tensor of 2 images concatenated along channel dimension
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            scale_invariant (`str`, *optional*, defaults to `True`):
                Flag of scale-invariant prediction, if True, scale will be adjusted from the raw prediction.
            shift_invariant (`str`, *optional*, defaults to `True`):
                Flag of shift-invariant prediction, if True, shift will be adjusted from the raw prediction, if False, near plane will be fixed at 0m.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """

        rgb = input_image
        rgb_next = next_input_image
        input_size = rgb.shape

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )
            if rgb_next is not None:
                rgb_next = resize_max_res(
                    rgb_next,
                    max_edge_resolution=processing_res,
                    resample_method=resample_method,
                )

        # Assert images are not normalized yet (should be in [0,255] range)
        assert rgb.min() >= 0 and rgb.max() <= 255, f"Input images should be in [0,255] range, got min={rgb.min()}, max={rgb.max()}"
        if rgb_next is not None:
            assert rgb_next.min() >= 0 and rgb_next.max() <= 255, f"Input images should be in [0,255] range, got min={rgb_next.min()}, max={rgb_next.max()}"

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        if rgb_next is not None:
            rgb_next_norm: torch.Tensor = rgb_next / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
            assert rgb_next_norm.min() >= -1.0 and rgb_next_norm.max() <= 1.0
        else:
            rgb_next_norm = None


        output_pred_ts = self.single_infer(
            rgb_norm=rgb_norm,
            rgb_next_norm=rgb_next_norm,
            num_inference_steps=denoising_steps,
            show_pbar=show_progress_bar,
            generator=generator,
            output_type=output_type,
            exclude_mainstream_output_type=exclude_mainstream_output_type,
            task_output_types=task_output_types,
        )

        # Resize back to original resolution
        if match_input_res:
            output_pred_ts = resize(
                output_pred_ts,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )

        # Convert to numpy
        output_pred_ts = output_pred_ts.squeeze()
        output_pred = output_pred_ts.cpu().numpy()

        if output_type == "albedo":
            assert output_pred.min() >= -1.0 and output_pred.max() <= 1.0
            output_pred = (output_pred + 1.0) / 2.0
            output = StableMTLAlbedoOutput(
                albedo_np=output_pred,
            )

        elif output_type == "shading":
            assert output_pred.min() >= -1.0 and output_pred.max() <= 1.0
            output_pred = (output_pred + 1.0) / 2.0
            output = StableMTLShadingOutput(
                shading_np=output_pred,
            )

        elif output_type == "depth":
            assert output_pred.min() >= -1.0 and output_pred.max() <= 1.0
            # shift to [0, 1]
            output_pred = (output_pred + 1.0) / 2.0

            # Colorize
            if color_map is not None:
                depth_colored = colorize_depth_maps(
                    output_pred, 0, 1, cmap=color_map
                ).squeeze()  # [3, H, W], value in (0, 1)
                depth_colored = (depth_colored * 255).astype(np.uint8)
                depth_colored_hwc = chw2hwc(depth_colored)
                depth_colored_img = Image.fromarray(depth_colored_hwc)
            else:
                depth_colored_img = None

            output = StableMTLDepthOutput(
                depth_np=output_pred,
                depth_colored=depth_colored_img,
            )
        elif output_type == "normal":
            assert output_pred.min() >= -1.0 and output_pred.max() <= 1.0
            # output_pred in [0, 1]
            output_pred_norm = np.linalg.norm(output_pred, axis=0, keepdims=True)
            output_pred_norm[output_pred_norm == 0] = 1.0
            output_pred = output_pred / output_pred_norm

            normal_colored = (1 - output_pred) / 2
            normal_colored = (normal_colored * 255).astype(np.uint8)
            normal_colored_hwc = chw2hwc(normal_colored)
            normal_colored_img = Image.fromarray(normal_colored_hwc)
            output = StableMTLNormalOutput(
                normal_np=output_pred,
                normal_colored=normal_colored_img,
            )
        elif output_type == "optical_flow":
            output = StableMTLOpticalFlowOutput(
                optical_flow_np=output_pred, # [-1.0, 1.0]
            )
        elif output_type == "scene_flow":
            output = StableMTLSceneFlowOutput(
                scene_flow_np=output_pred, # [-1.0, 1.0]
            )
        elif output_type == "semantic":
            class_color_embeddings = self.semantic_encoder.class_color_embeddings
            class_color_embeddings = torch.from_numpy(class_color_embeddings).type_as(output_pred_ts)
            class_color_embeddings = class_color_embeddings / 255.0 * 2.0 - 1.0
            output_pred_ts = output_pred_ts.permute(1, 2, 0).reshape(-1, 3)
            distances = torch.cdist(output_pred_ts, class_color_embeddings)
            semantic_class_id = torch.argmin(distances, dim=1)
            semantic_class_id = semantic_class_id.reshape(output_pred.shape[1:])

            output = StableMTLSemsegOutput(
                semantic_class_id=semantic_class_id.cpu().numpy(),
                class_color_visualizes=self.semantic_encoder.class_color_visualizes
            )
        else:
            raise ValueError(f"Unknown output type: {output_type}")

        return output


    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                pass
                # logging.warning(
                #     f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                # )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_text(self, prompt):
        """
        Encode text embedding for empty prompt
        """
        text_inputs = self.tokenizer(
            prompt,
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        text_embed = self.text_encoder(text_input_ids)[0]
        return text_embed


    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0]#.to(self.dtype)


    def encode_rgb_latent(self, output_type, rgb_norm, rgb_next_norm):
        assert output_type in set(["optical_flow", "scene_flow",
                                   "depth", "normal", "semantic",
                                   "albedo", "shading"]), f"Unknown output type: {output_type}"
        rgb_in = self.encode_rgb(rgb_norm)
        rgb_next_in = None
        if output_type in set(["optical_flow", "scene_flow"]) and rgb_next_norm is not None:
            rgb_next_in = self.encode_rgb(rgb_next_norm)
        else:
            if self.encode_rgb_model == "duplicate":
                rgb_next_in = rgb_in
            elif self.encode_rgb_model == "zero":
                rgb_next_in = torch.zeros_like(rgb_in)
        
        if self.encode_rgb_model == "avg":
            if rgb_next_in is None:
                rgb_latent = rgb_in
            else:
                rgb_latent = (rgb_in + rgb_next_in) / 2.0
        else:
            rgb_latent = torch.cat([rgb_in, rgb_next_in], dim=1)
     
        rgb_latent = rgb_latent.unsqueeze(1) # [B, 1, c, h, w]
        rgb_latent = rearrange(rgb_latent, 'b f c h w -> b c f h w') # [B, c, 1, h, w]

        return rgb_latent

    def create_task_input_stream(self, output_types, rgb_norm, rgb_next_norm):
        rgb_latents = []
        for output_type in output_types:
            rgb_latent = self.encode_rgb_latent(output_type, rgb_norm, rgb_next_norm)
            rgb_latents.append(rgb_latent)
        rgb_latent = torch.cat(rgb_latents, dim=2)

        return rgb_latent


    def create_text_condition(self, output_types, batch_size):
        text_condition = []
        for output_type in output_types:
            output_type = output_type.replace("_", " ")
            text_condition.append(output_type)
        text_embed = self.encode_text(text_condition).detach().clone().to(self.device) # (n_task, n, 1024)
        text_embed = text_embed.unsqueeze(0).expand(batch_size, -1, -1, -1)
        text_embed = rearrange(text_embed, 'b f n c -> (b f) n c')
        return text_embed


    def create_task_feats(self, rgb_norm, rgb_next_norm, timesteps, output_type,
                          task_output_types, rand_num_generator, 
                          exclude_mainstream_output_type,
                          drop_ratio=0.0):
        if not hasattr(self, "unet_child") or self.unet_child is None:
            return None, None

        # assert len(task_output_types) == 7, f"Expected 7 output types, got {len(task_output_types)}"
        if exclude_mainstream_output_type:
            task_output_types = [t for t in task_output_types if t != output_type]

        if drop_ratio > 0.0:
            rand_num = random.random()
            if rand_num < drop_ratio:
                task_output_types = np.random.choice(task_output_types, size=len(task_output_types) - 1, replace=False)

        batch_size = rgb_norm.shape[0]
        list_task_feats = [{} for _ in range(16)]
        unet_child_outputs = []
        with torch.no_grad():
            for task_output_type in task_output_types:
                child_text_embed = self.create_text_condition([task_output_type], batch_size)
                rgb_latent = self.encode_rgb_latent(task_output_type,
                                                    rgb_norm=rgb_norm,
                                                    rgb_next_norm=rgb_next_norm)
                if self.input_noise == "deterministic":
                    noise = torch.zeros_like(rgb_latent[:, :4, :, :, :])
                elif self.input_noise == "random":
                    noise = torch.randn(rgb_latent[:, :4, :, :, :].shape, device=self.device,
                                generator=rand_num_generator)
                else:
                    raise ValueError(f"Unknown input noise: {self.input_noise}")
                cat_latents = torch.cat([rgb_latent, noise], dim=1)
                unet_child_output, ret_task_feats = self.unet_child(
                    cat_latents, timesteps, child_text_embed
                )
                unet_child_outputs.append(unet_child_output.sample)
                for i, task_value in enumerate(ret_task_feats):
                    list_task_feats[i][task_output_type] = task_value

        return unet_child_outputs, list_task_feats


    @torch.no_grad()
    def single_infer(
        self,
        rgb_norm: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        output_type: str,
        exclude_mainstream_output_type: bool,
        rgb_next_norm: Optional[torch.Tensor] = None,
        task_output_types: List[str] = [],
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_norm = rgb_norm.to(device)
        if rgb_next_norm is not None:
            rgb_next_norm = rgb_next_norm.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = [999] # set to a fixed timestep like in lotus

        rgb_latent = self.encode_rgb_latent(output_type, rgb_norm, rgb_next_norm)
        noise = torch.randn(rgb_latent[:, :4, :, :, :].shape, device=device, generator=generator)

        if self.input_noise == "deterministic":
            output_latent = torch.zeros_like(noise)
        elif self.input_noise == "random":
            output_latent = noise
        else:
            raise ValueError(f"Unknown input noise: {self.input_noise}")

        batch_size = rgb_latent.shape[0]
        text_embed = self.create_text_condition([output_type], batch_size)


        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        # ===== Single timestep inference ======
        t = timesteps[0]

        unet_input = torch.cat(
            [rgb_latent, output_latent], dim=1
        )  # this order is important

        unet_child_output, task_feats = self.create_task_feats(rgb_norm, rgb_next_norm,
                                                t,
                                                output_type=output_type,
                                                task_output_types=task_output_types,
                                                rand_num_generator=generator,
                                                drop_ratio=0.0,
                                                exclude_mainstream_output_type=exclude_mainstream_output_type)

        # predict the noise residual
        unet_output, _ = self.unet(
            unet_input, t, encoder_hidden_states=text_embed, task_feats=task_feats, output_type=output_type
        )  # [B, 4, n_tasks, h, w]
        x0_pred = unet_output.sample.squeeze(2) # remove the task dimension, currently always 1
        output = self.decode_output(x0_pred, output_type)
        # clip prediction
        output = torch.clip(output, -1.0, 1.0)


        return output


    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_output(self, latent: torch.Tensor, output_type: str) -> torch.Tensor:
        """
        Decode task latent into the corresponding output map.

        Args:
            latent (`torch.Tensor`):
                Task latent to be decoded.
            output_type (`str`):
                Type of output to decode (e.g., "depth", "normal", "semantic", etc.).

        Returns:
            `torch.Tensor`: Decoded output map appropriate for the specified task.
        """
        # Scale latent
        latent = latent / self.latent_scale_factor
        # Decode
        z = self.vae.post_quant_conv(latent)
        stacked = self.vae.decoder(z)

        if output_type in ["depth", "shading"]:
            # Mean of output channels for single-channel outputs
            output = stacked.mean(dim=1, keepdim=True)
            return output
        elif output_type in ["normal", "semantic", "rgb", "scene_flow", "albedo"]:
            # Multi-channel outputs returned as is
            return stacked
        elif output_type == "optical_flow":
            # Optical flow uses only the first two channels
            return stacked[:, :2]
        else:
            raise ValueError(f"Unknown output type: {output_type}")


 