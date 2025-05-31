
import logging
import os
import random
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.utils import BaseOutput
from omegaconf import OmegaConf
from PIL import Image
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.base_mtl_dataset import DatasetConst
from src.stablemtl_pipeline import StableMTLPipeline
from src.util import metric as metric_depth
from src.util import metric_normal
from src.util.alignment import (
    align_depth_least_square,
    align_flow_least_square,
    align_flow_norm_least_square,
)
from src.util.data_loader import skip_first_batches
from src.util.logging_util import eval_dic_to_text, tb_logger
from src.util.loss import MovingAverageLossWeighter, compute_grad_norm, get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.metric_albedo_and_shading import AlbedoAndShadingMetrics, match_scale
from src.util.metric_optical_flow import OpticalFlowMetrics, SceneFlowMetrics
from src.util.metric_semantic import SemanticMetrics
from src.util.seeding import generate_seed_sequence
from src.util.visualizer import (
    tone_map,
    visualize_depth,
    visualize_optical_flow,
    visualize_optical_flow_pred_only,
    visualize_scene_flow_pred_only,
    visualize_semantic,
    visualize_semantic_pred_only,
)


class StableMTLTrainer:
    """Trainer class for StableMTL model.

    This class handles training, evaluation, visualization, and checkpoint management
    for the StableMTL pipeline, which is a multi-task learning framework based on
    diffusion models.
    """

    def __init__(
        self,
        cfg: OmegaConf,
        model: StableMTLPipeline,
        train_dataloader: DataLoader,
        accelerator,
        base_ckpt_dir: str,
        out_dir_ckpt: str,
        out_dir_eval: str,
        out_dir_vis: str,
        accumulation_steps: int = 1,
        val_dataloaders: Optional[List[DataLoader]] = None,
        vis_dataloaders: Optional[List[DataLoader]] = None,
        input_noise: str = "random",
        no_lr_scheduler: bool = False,
        exclude_mainstream_output_type: bool = True,
    ):
        """Initialize the StableMTL trainer.

        Args:
            cfg: Configuration object.
            model: StableMTL pipeline model.
            train_dataloader: DataLoader for training data.
            accelerator: Accelerator for distributed training.
            base_ckpt_dir: Base directory for checkpoints.
            out_dir_ckpt: Output directory for saving checkpoints.
            out_dir_eval: Output directory for evaluation results.
            out_dir_vis: Output directory for visualizations.
            accumulation_steps: Number of gradient accumulation steps.
            val_dataloaders: List of validation dataloaders.
            vis_dataloaders: List of visualization dataloaders.
            input_noise: Type of input noise ("random" or other).
            no_lr_scheduler: Whether to disable learning rate scheduler.
            exclude_mainstream_output_type: Whether to exclude mainstream output type.
        """
        self.cfg: OmegaConf = cfg
        self.model: StableMTLPipeline = model
        self.accelerator = accelerator
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps
        self.input_noise: str = input_noise
        self.no_lr_scheduler: bool = no_lr_scheduler
        self.exclude_mainstream_output_type: bool = exclude_mainstream_output_type
        self.effective_iter = 0

        # Optimizer (should be defined after input layer is adapted)
        lr = self.cfg.lr
        self.optimizer = Adam(self.model.unet.parameters(), lr=lr)

        # LR scheduler
        self.lr_scheduler = None
        if not self.no_lr_scheduler:
            lr_func = IterExponential(
                total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter * self.accumulation_steps,
                final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
                warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps * self.accumulation_steps,
            )
            self.lr_scheduler = LambdaLR(
                optimizer=self.optimizer, lr_lambda=lr_func)

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name,
                             **self.cfg.loss.kwargs)

        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            os.path.join(
                base_ckpt_dir, cfg.trainer.training_noise_scheduler.pretrained_path
            ),
            subfolder="scheduler",
            prediction_type=cfg.trainer.training_noise_scheduler.prediction_type,
            num_train_timesteps=cfg.trainer.training_noise_scheduler.num_train_timesteps,
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type

        logging.info(f"Prediction type: {self.prediction_type}")
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        self.output_types = cfg.trainer.output_types
        self.main_val_metrics = {}
        self.main_val_metric_goals = {}

        if self.accelerator.is_main_process:
            # Eval metrics
            self.depth_metric_funcs = [
                getattr(metric_depth, _met) for _met in cfg.eval.eval_depth_metrics]
            self.normal_metric_funcs = [
                getattr(metric_normal, _met) for _met in cfg.eval.eval_normal_metrics]
            loss_list = []
            for output_type in self.output_types:
                loss_list.append(f"loss_{output_type}")
            self.train_metrics = MetricTracker(*loss_list)
            self.val_depth_metrics = MetricTracker(
                *[m.__name__ for m in self.depth_metric_funcs])
            self.val_normal_metrics = MetricTracker(
                *[m.__name__ for m in self.normal_metric_funcs])
            self.val_semantic_metrics = SemanticMetrics(n_classes=8)
            self.val_optical_flow_metrics = OpticalFlowMetrics()
            self.val_scene_flow_metrics = SceneFlowMetrics()
            self.val_albedo_metrics = AlbedoAndShadingMetrics()
            self.val_shading_metrics = AlbedoAndShadingMetrics()

        self.best_metric = 1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps

        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = (
            []
        )  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

        self.loss_weighter = MovingAverageLossWeighter(
            loss_names=self.output_types)

    def downsample_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """Downsample a mask by max pooling.

        Args:
            valid_mask: Input mask tensor.

        Returns:
            Downsampled mask tensor.
        """
        invalid_mask = ~valid_mask
        valid_mask_down = ~torch.max_pool2d(
            invalid_mask.float(), 8, 8
        ).bool()
        valid_mask_down = valid_mask_down.repeat((1, 4, 1, 1))
        return valid_mask_down

    def train(self, t_end: Optional[int] = None):
        """Train the model.

        Args:
            t_end: Optional ending timestep.
        """
        if self.accelerator.is_main_process:
            logging.info("Start training")

        device = self.accelerator.device

        self.accelerator.wait_for_everyone()

        self.model.vae.to(device)
        self.model.text_encoder.to(device)

        if self.accelerator.is_main_process:
            self.train_metrics.reset()

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.unet.train()
                self.n_batch_in_epoch += 1

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                    random.seed(local_seed)
                else:
                    rand_num_generator = None

                # Get data
                rgb_norm = batch["rgb_norm"].to(device)
                rgb_next_norm = batch["rgb_next_norm"].to(
                    device) if "rgb_next_norm" in batch else None

                batch_size = rgb_norm.shape[0]

                with torch.no_grad():
                    # Encode image
                    output_type = batch[DatasetConst.OUTPUT_TYPE_FIELD][0]
                    rgb_latent = self.model.encode_rgb_latent(output_type,
                                                              rgb_norm=rgb_norm,
                                                              rgb_next_norm=rgb_next_norm)
                    valid_mask = batch[DatasetConst.VALID_MASK_FIELD].to(
                        device)
                    valid_mask_down = self.downsample_mask(valid_mask)

                    gt_output = batch[DatasetConst.OUTPUT_FIELD].to(device)
                    gt_output_latent = self.encode_output(
                        gt_output, output_type=output_type)

                    text_embed = self.model.create_text_condition(
                        [output_type], batch_size)

                    timesteps = torch.ones(
                        (batch_size,), device=device, dtype=torch.long) * 999
                    noise = torch.randn(
                        rgb_latent[:, :4, :, :, :].shape, device=device, generator=rand_num_generator)

                    if self.input_noise == "deterministic":
                        noisy_latents = torch.zeros_like(noise)
                    elif self.input_noise == "random":
                        noisy_latents = noise
                    else:
                        raise ValueError(
                            f"Unknown input noise: {self.input_noise}")

                    cat_latents = torch.cat(
                        [rgb_latent, noisy_latents], dim=1).float()

                    unet_child_outputs, task_feats = self.model.create_task_feats(rgb_norm, rgb_next_norm,
                                                                                  timesteps,
                                                                                  output_type=output_type,
                                                                                  task_output_types=self.output_types,
                                                                                  rand_num_generator=rand_num_generator,
                                                                                  drop_ratio=0.0,
                                                                                  exclude_mainstream_output_type=self.exclude_mainstream_output_type)

                with self.accelerator.accumulate(self.model.unet):
                    # Forward pass
                    unet_output, ret_samples = self.model.unet(
                        cat_latents, timesteps, text_embed,
                        task_feats=task_feats, output_type=output_type
                    )
                    model_pred = unet_output.sample.squeeze(2)
                    if torch.isnan(model_pred).any():
                        logging.warning("model_pred contains NaN.")

                    # Get target for loss
                    if "sample" == self.prediction_type:
                        target = gt_output_latent
                    elif "epsilon" == self.prediction_type:
                        target = noise
                    elif "v_prediction" == self.prediction_type:
                        target = self.training_noise_scheduler.get_velocity(
                            gt_output_latent, noise, timesteps
                        )
                    else:
                        raise ValueError(
                            f"Unknown prediction type {self.prediction_type}")

                    loss = self.loss(
                        model_pred[valid_mask_down].float(),
                        target[valid_mask_down].float(),
                    ).mean()

                    try:
                        # Scale loss and backward pass
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            grad_norm, std_norm = compute_grad_norm(
                                self.model.unet)
                            self.accelerator.clip_grad_norm_(
                                self.model.unet.parameters(), 5.0)
                    except Exception as e:
                        logging.error(f"Error during backward pass: {e}")
                        raise e

                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                self.iter += 1
                self.effective_iter = self.iter // self.gradient_accumulation_steps
                # Logging and checkpointing
                if self.iter % self.gradient_accumulation_steps == 0:
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:

                        if loss is not None:  # Only log if loss exists
                            self.train_metrics.update(
                                f"loss_{output_type}", loss.item())

                            accumulated_loss = 0.0
                            train_metrics_result = self.train_metrics.result()
                            for _, v in train_metrics_result.items():
                                accumulated_loss += v

                            tb_logger.log_dic({
                                f"train_grad_norm_mean/{output_type}": grad_norm,
                                f"train_grad_norm_std/{output_type}": std_norm,
                            }, global_step=self.effective_iter)

                            for k, v in self.train_metrics.result().items():
                                if v != 0:
                                    tb_logger.log_dic(
                                        {f"train/{k}": v}, global_step=self.effective_iter
                                    )
                            # Log learning rate - works with or without scheduler
                            if self.lr_scheduler is not None:
                                current_lr = self.lr_scheduler.get_last_lr()[0]
                            else:
                                # Get learning rate directly from optimizer when no scheduler
                                current_lr = self.optimizer.param_groups[0]['lr']

                            tb_logger.writer.add_scalar(
                                "lr",
                                current_lr,
                                global_step=self.effective_iter,
                            )
                            tb_logger.writer.add_scalar(
                                "n_batch_in_epoch",
                                self.n_batch_in_epoch,
                                global_step=self.effective_iter,
                            )
                            logging.info(
                                f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                            )
                            self.train_metrics.reset()

                        # Per-step callback
                        self._train_step_callback()

                if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(), save_train_state=True
                    )
                    logging.info("Training ended.")
                    return
            # Epoch end
            self.n_batch_in_epoch = 0

    def eval(self):
        """Evaluate the model.

        Args:
            mode: Evaluation mode.
            infer_budget_task: Task for inference budget evaluation.
        """

        if self.accelerator.is_main_process:
            logging.info("Start evaluation")

        device = self.accelerator.device

        self.model.vae.to(device)
        self.model.text_encoder.to(device)

        components_to_prepare = [
            self.model.unet,
        ]

        # Add unet_child to preparation if it exists
        if self.model.unet_child is not None:
            components_to_prepare.insert(1, self.model.unet_child)
            prepared_components = self.accelerator.prepare(*components_to_prepare)
            self.model.unet, self.model.unet_child = prepared_components
        else:
            prepared_components = self.accelerator.prepare(*components_to_prepare)
            self.model.unet = prepared_components

        self.model.unet.eval()

        self.validate(tb_log=False)
        self.visualize(out_dir="evaluation")
        
    
    def encode_output(self, output_in: torch.Tensor, output_type: str) -> torch.Tensor:
        """Encode output tensor based on output type.

        Args:
            output_in: Input tensor.
            output_type: Type of output.

        Returns:
            Encoded output tensor.
        """
        # if output_type == "scene_flow":
        #     output_latent = self.model.encode_rgb(output_in)
        if output_type == "optical_flow":
            stacked = torch.cat([output_in, output_in[:, :1, :, :]], dim=1)
            output_latent = self.model.encode_rgb(stacked)
        elif output_type in ["normal", "semantic", "albedo", "scene_flow"]:

            output_latent = self.model.encode_rgb(output_in)
        elif output_type in ["depth", "shading"]:
            stacked = self.stack_depth_images(output_in)  # range depth [-1, 1]
            # encode using VAE encoder
            output_latent = self.model.encode_rgb(stacked)
        else:
            raise ValueError(f"Unknown output type: {output_type}")
        return output_latent

    @staticmethod
    @staticmethod
    def stack_depth_images(depth_in: torch.Tensor) -> torch.Tensor:
        """Stack depth images for visualization.

        Args:
            depth_in: Input depth tensor.

        Returns:
            Stacked depth tensor.
        """
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1)
            stacked = depth_in.repeat(1, 3, 1, 1)
        return stacked

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=True
            )

        _is_latest_saved = False

        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            # flag to do evaluation in resume run if validation is not finished
            self.in_evaluation = True
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True

            # Validate all output types
            self.validate()

            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            # Visualize all output types
            self.visualize()

    def validate(self, tb_log: bool = True):
        """Validate the model.

        Args:
            tb_log: Whether to log to TensorBoard.
        """
        self.model.unet.eval()
        for i, val_loader in enumerate(self.val_loaders):

            val_metrics = self.validate_single_dataset(
                data_loader=val_loader,
            )

            val_dataset_name = val_loader.dataset.disp_name
            for output_type in val_metrics:
                val_metric_result = val_metrics[output_type].result()

                logging.info(
                    f"Iter {self.effective_iter}. Validation metrics on `{output_type}/{val_dataset_name}`: {val_metric_result}"
                )
                if tb_log:
                    tb_logger.log_dic(
                        {f"val_{output_type}_{val_dataset_name}/{k}": v for k,
                            v in val_metric_result.items()},
                        global_step=self.effective_iter,
                    )
                # save to file
                eval_text = eval_dic_to_text(
                    val_metrics=val_metric_result,
                    dataset_name=val_dataset_name,
                    sample_list_path=val_loader.dataset.filename_ls_path,
                )
                print(eval_text)

    def visualize(self, out_dir: Optional[str] = None):
        """Generate visualizations.

        Args:
            out_dir: Output directory for visualizations.
        """
        if self.vis_loaders is None:
            return
        self.model.unet.eval()
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            if out_dir is None:
                vis_out_dir = os.path.join(
                    self.out_dir_vis, self._get_backup_ckpt_name(
                    ), f"{vis_dataset_name}"
                )
            else:
                vis_out_dir = os.path.join(self.out_dir_vis, out_dir)

            os.makedirs(vis_out_dir, exist_ok=True)
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                save_to_dir=vis_out_dir,
            )


    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        save_to_dir: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        """Validate model on a single dataset.

        Args:
            data_loader: DataLoader for the dataset.
            save_to_dir: Directory to save results.
            mode: Validation mode.
        """
        all_val_metrics = {
            "depth": self.val_depth_metrics,
            "normal": self.val_normal_metrics,
            "semantic": self.val_semantic_metrics,
            "optical_flow": self.val_optical_flow_metrics,
            "scene_flow": self.val_scene_flow_metrics,
            "albedo": self.val_albedo_metrics,
            "shading": self.val_shading_metrics,
        }

        val_metrics = {}
        # self.model.to(self.accelerator.device)
        per_task_save_to_dir = {}
        if mode == "draw_qualitative_results" or mode == "draw_video_results":
            output_types_to_process = self.output_types
        else:
            output_types_to_process = data_loader.dataset.output_type

        for output_type in output_types_to_process:
            if output_type in self.output_types:
                if save_to_dir is not None:
                    if mode == "draw_video_results":
                        per_task_save_to_dir[output_type] = os.path.join(
                            save_to_dir, output_type)
                        os.makedirs(
                            per_task_save_to_dir[output_type], exist_ok=True)
                    else:
                        per_task_save_to_dir[output_type] = os.path.join(
                            save_to_dir)
                        os.makedirs(
                            per_task_save_to_dir[output_type], exist_ok=True)

                val_metrics[output_type] = all_val_metrics[output_type]
                val_metrics[output_type].reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        # val_init_seed = 1
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))
        action = "evaluating" if save_to_dir is None else "visualizing"
        previous_flow = None
        for i, batch in enumerate(
            tqdm(data_loader,
                 desc=f"{action} on {data_loader.dataset.disp_name}"),
            start=1,
        ):

            assert 1 == data_loader.batch_size

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.accelerator.device)
                generator.manual_seed(seed)

            for output_type in output_types_to_process:
                if output_type not in self.output_types:
                    continue
                # Read input image
                rgb_int = batch["rgb_int"].clone()  # [B, 3, H, W]

                rgb_next_int = batch["rgb_next_int"].clone(
                ) if "rgb_next_int" in batch else None

                if mode is not None:
                    # Save RGB images
                    img_name = batch["rgb_relative_path"][0].replace("/", "_")
                    base_name, ext = os.path.splitext(img_name)
                    if not ext:
                        ext = ".jpg"  # Default to jpg if no extension

                    # Save current RGB image
                    if mode == "draw_video_results":
                        rgb_save_path = os.path.join(
                            save_to_dir, "frame1", f"{base_name}{ext}")
                        os.makedirs(os.path.dirname(
                            rgb_save_path), exist_ok=True)
                    else:
                        rgb_save_path = os.path.join(
                            save_to_dir, f"{base_name}{ext}")
                    rgb_vis = (rgb_int.squeeze(0).detach().cpu(
                    ).numpy().transpose(1, 2, 0)).astype(np.uint8)
                    Image.fromarray(rgb_vis).save(rgb_save_path)
                    print(f"Save RGB to {rgb_save_path}")

                    # Save next RGB image if available
                    if rgb_next_int is not None:
                        if mode == "draw_video_results":
                            rgb_next_save_path = os.path.join(
                                save_to_dir, "frame2", f"{base_name}{ext}")
                            os.makedirs(os.path.dirname(
                                rgb_next_save_path), exist_ok=True)
                        else:
                            rgb_next_save_path = os.path.join(
                                save_to_dir, f"{base_name}_next{ext}")
                        rgb_next_vis = (rgb_next_int.squeeze(0).detach(
                        ).cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
                        Image.fromarray(rgb_next_vis).save(rgb_next_save_path)
                        print(f"Save RGB next to {rgb_next_save_path}")

                # Predict depth
                pipe_out: BaseOutput = self.model(
                    input_image=rgb_int,
                    next_input_image=rgb_next_int,
                    denoising_steps=self.cfg.validation.denoising_steps,
                    ensemble_size=self.cfg.validation.ensemble_size,
                    processing_res=self.cfg.validation.processing_res,
                    match_input_res=self.cfg.validation.match_input_res,
                    generator=generator,
                    batch_size=1,  # use batch size 1 to increase reproducibility
                    color_map=None,
                    show_progress_bar=False,
                    resample_method=self.cfg.validation.resample_method,
                    output_type=output_type,
                    task_output_types=self.output_types,
                    exclude_mainstream_output_type=self.exclude_mainstream_output_type,
                )

                if output_type == "albedo":
                    # (1, 3, H, W)
                    albedo_pred = pipe_out.albedo_np[np.newaxis, ...]
                    if "albedo" in batch:
                        albedo_gt = batch["albedo"].detach(
                        ).cpu().numpy()  # (1, 3, H, W)
                        valid_mask = batch["albedo_valid_mask"].detach(
                        ).cpu().numpy()  # (1, 1, H, W)
                        val_metrics[output_type].update(
                            albedo_pred, albedo_gt, valid_mask)

                    if output_type in per_task_save_to_dir:
                        img_name = batch["rgb_relative_path"][0].replace(
                            "/", "_")
                        # Extract base name without extension and determine file extension
                        base_name, ext = os.path.splitext(img_name)
                        if not ext:
                            ext = ".jpg"  # Default to jpg if no extension

                        if mode == "draw_video_results":
                            save_name = f"{base_name}{ext}"
                        else:
                            save_name = f"{base_name}_albedo{ext}"
                        save_path = os.path.join(
                            per_task_save_to_dir[output_type], save_name)

                        # Convert albedo from (1,3,H,W) to (H,W,3) and scale to [0,255]
                        albedo_pred = albedo_pred.squeeze(
                            0).transpose(1, 2, 0)  # (H,W,3)

                        if "albedo" in batch:
                            albedo_gt = albedo_gt.squeeze(
                                0).transpose(1, 2, 0)  # (H,W,3)
                            valid_mask = valid_mask.squeeze(0).transpose(
                                1, 2, 0).astype(bool)  # (H,W,1)
                            albedo_vis = (albedo_gt * 255).astype(np.uint8)
                            Image.fromarray(albedo_vis).save(
                                save_path.replace(ext, f"_gt{ext}"))
                            print(
                                f"Save albedo gt to {save_path.replace(ext, f'_gt{ext}')}")

                            scale = match_scale(
                                albedo_pred, albedo_gt, valid_mask)
                            albedo_pred = (albedo_pred * scale).clip(0, 1)

                        albedo_vis = (albedo_pred * 255).astype(np.uint8)
                        Image.fromarray(albedo_vis).save(save_path)
                        print(f"Save albedo to {save_path}")

                elif output_type == "shading":
                    # (1, 1, H, W)
                    shading_pred = pipe_out.shading_np[np.newaxis,
                                                       np.newaxis, ...]
                    if "shading" in batch:
                        shading_gt = batch["shading"].detach(
                        ).cpu().numpy()  # (1, 3, H, W)
                        valid_mask = batch["shading_valid_mask"].detach(
                        ).cpu().numpy()  # (1, 1, H, W)
                        val_metrics[output_type].update(
                            shading_pred, shading_gt, valid_mask)

                    if output_type in per_task_save_to_dir:
                        img_name = batch["rgb_relative_path"][0].replace(
                            "/", "_")
                        # Extract base name without extension and determine file extension
                        base_name, ext = os.path.splitext(img_name)
                        if not ext:
                            ext = ".jpg"  # Default to jpg if no extension

                        if mode == "draw_video_results":
                            save_name = f"{base_name}{ext}"
                        else:
                            save_name = f"{base_name}_shading{ext}"
                        save_path = os.path.join(
                            per_task_save_to_dir[output_type], save_name)

                        # Convert shading from (1,3,H,W) to (H,W,3)
                        shading_pred = shading_pred.squeeze(
                            0).transpose(1, 2, 0)  # (H,W,3)

                        if "shading" in batch:
                            shading_gt = shading_gt.squeeze(
                                0).transpose(1, 2, 0)        # (H,W,3)
                            valid_mask = valid_mask.squeeze(0).transpose(
                                1, 2, 0).astype(bool)  # (H,W,1)

                            # Match scale between predicted shading and ground truth
                            scale = match_scale(
                                shading_pred, shading_gt, valid_mask)
                            shading_pred = (shading_pred * scale).clip(0, 1)

                            # Save the ground truth shading image (grayscale)
                            shading_vis = (shading_gt * 255).astype(np.uint8)
                            Image.fromarray(shading_vis.squeeze()).convert(
                                'L').save(save_path.replace(ext, f"_gt{ext}"))
                            print(
                                f"Save shading gt to {save_path.replace(ext, f'_gt{ext}')}")
                            shading_pred_tone_mapped = None
                        else:
                            shading_pred_tone_mapped = tone_map(shading_pred)

                       # Save the predicted shading image (grayscale)
                        shading_vis = (shading_pred * 255).astype(np.uint8)
                        Image.fromarray(shading_vis.squeeze()).convert(
                            'L').save(save_path)
                        print(f"Save shading to {save_path}")

                elif output_type == "semantic":

                    class_color_visualizes = pipe_out.class_color_visualizes
                    semantic_class_id_pred = pipe_out.semantic_class_id[None, ...]

                    if "semantic_class_id" in batch:
                        semantic_class_id_gt = batch["semantic_class_id"]
                        valid_mask = batch["semantic_valid_mask"]
                        assert semantic_class_id_gt.shape[0] == 1, "Batch size must be 1 for evaluation"
                        semantic_class_id_gt = semantic_class_id_gt.detach().cpu().numpy()
                        valid_mask = valid_mask.detach().cpu().numpy()

                        val_metrics[output_type].update(semantic_class_id_gt.squeeze(
                            1), semantic_class_id_pred, valid_mask.squeeze(1))

                    if output_type in per_task_save_to_dir:
                        img_name = batch["rgb_relative_path"][0].replace(
                            "/", "_")
                        base_name, ext = os.path.splitext(img_name)

                        if mode == "draw_video_results":
                            save_name = f"{base_name}{ext}"
                        else:
                            save_name = f"{base_name}_semantic{ext}"
                        save_path = os.path.join(
                            per_task_save_to_dir[output_type], save_name)
                        if "semantic_class_id_gt" in batch:
                            gt_save_path = save_path.replace(ext, f"_gt{ext}")
                            visualize_semantic_pred_only(
                                semantic_class_id_gt.squeeze(), class_color_visualizes, gt_save_path)

                        visualize_semantic_pred_only(
                            semantic_class_id_pred.squeeze(), class_color_visualizes, save_path)

                elif output_type == "normal":
                    normal_pred: np.ndarray = pipe_out.normal_np
                    if "normal" in batch:
                        normal_gt_ts = batch["normal"].to(
                            self.accelerator.device)
                        valid_mask_ts = batch["normal_valid_mask"].to(
                            self.accelerator.device)

                        # Evaluate
                        sample_metric = []
                        normal_pred_ts = torch.from_numpy(normal_pred).to(
                            self.accelerator.device).unsqueeze(0)

                        for met_func in self.normal_metric_funcs:
                            _metric_name = met_func.__name__

                            _metric = met_func(
                                normal_pred_ts, normal_gt_ts, valid_mask_ts).item()
                            sample_metric.append(_metric.__str__())
                            val_metrics[output_type].update(
                                _metric_name, _metric)

                    if output_type in per_task_save_to_dir:
                        img_name = batch["rgb_relative_path"][0].replace(
                            "/", "_")
                        base_name, ext = os.path.splitext(img_name)

                        if mode == "draw_video_results":
                            save_name = f"{base_name}{ext}"
                        else:
                            save_name = f"{base_name}_normal{ext}"
                        save_path = os.path.join(
                            per_task_save_to_dir[output_type], save_name)

                        normal_vis = (1 + normal_pred) / 2.0  # 3, H, W
                        normal_vis = normal_vis.transpose(1, 2, 0)  # H, W, 3
                        plt.imsave(save_path, normal_vis)

                        if "normal" in batch:
                            gt_save_path = save_path.replace(ext, f"_gt{ext}")
                            normal_gt = batch["normal"].squeeze(
                            ).detach().cpu().numpy()
                            normal_gt_vis = (1 + normal_gt) / 2.0
                            normal_gt_vis = normal_gt_vis.transpose(
                                1, 2, 0)  # H, W, 3
                            plt.imsave(gt_save_path, normal_gt_vis)

                elif output_type == "optical_flow":
                    optical_flow_pred = pipe_out.optical_flow_np
                    optical_flow_pred = torch.from_numpy(optical_flow_pred)

                    width, height = rgb_int.shape[3], rgb_int.shape[2]

                    if "optical_flow_raw" in batch:
                        optical_flow_gt = batch["optical_flow_raw"].detach().cpu()
                        valid_mask = batch[DatasetConst.VALID_MASK_FIELD].detach().cpu()

                        # Resize prediction to match ground truth dimensions if sizes differ

                        if optical_flow_pred.shape[-2:] != optical_flow_gt.shape[-2:]:
                            optical_flow_pred = F.interpolate(
                                optical_flow_pred.unsqueeze(0),
                                size=optical_flow_gt.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0)

                        norm_type = "hw"
                        # norm_type = "norm"
                        if norm_type == "hw":
                            optical_flow_pred_aligned = align_flow_least_square(
                                gt_arr=optical_flow_gt.numpy(),
                                pred_arr=optical_flow_pred.numpy(),
                                valid_mask_arr=valid_mask.numpy(),
                                return_scale_shift=False,
                            )
                        elif norm_type == "norm":
                            optical_flow_pred_aligned = align_flow_norm_least_square(
                                gt_arr=optical_flow_gt.numpy(),
                                pred_arr=optical_flow_pred.numpy(),
                                valid_mask_arr=valid_mask.numpy(),
                                return_scale_shift=False,
                            )
                        else:
                            raise ValueError(f"Unknown norm type: {norm_type}")

                        optical_flow_pred = torch.from_numpy(
                            optical_flow_pred_aligned).float()

                        val_metrics[output_type].update(optical_flow_preds=optical_flow_pred.unsqueeze(0),
                                                        optical_flow_gts=optical_flow_gt,
                                                        valid_masks=valid_mask)

                    if output_type in per_task_save_to_dir:
                        img_name = batch["rgb_relative_path"][0].replace(
                            "/", "_")
                        base_name, ext = os.path.splitext(img_name)
                        if mode == "draw_video_results":
                            save_name = f"{base_name}{ext}"
                        else:
                            save_name = f"{base_name}_opticalflow{ext}"
                        save_path = os.path.join(
                            per_task_save_to_dir[output_type], save_name)
                        if rgb_next_int is not None:
                            rgb_next_int = rgb_next_int.squeeze(
                                0).detach().cpu().numpy()
                        rgb_int = rgb_int.squeeze(0).detach().cpu().numpy()

                        visualize_optical_flow_pred_only(
                            flow_pred=optical_flow_pred.detach().cpu().numpy(),
                            png_save_path=save_path, max_flow=512)
                        if "optical_flow_raw" in batch:
                            visualize_optical_flow_pred_only(
                                flow_pred=optical_flow_gt.squeeze(
                                    0).detach().cpu().numpy(),
                                png_save_path=save_path.replace(ext, f"_gt{ext}"), max_flow=512)

                elif output_type == "scene_flow":
                    scene_flow_pred = pipe_out.scene_flow_np
                    scene_flow_pred = torch.from_numpy(scene_flow_pred)

                    if "scene_flow" in batch:
                        scene_flow_gt = batch["scene_flow"].detach().cpu()
                        valid_mask = batch["valid_mask"].detach().cpu()

                        norm_type = "hw"
                        # norm_type = "norm"
                        if norm_type == "hw":
                            scene_flow_pred_aligned = align_flow_least_square(
                                gt_arr=scene_flow_gt.numpy(),
                                pred_arr=scene_flow_pred.numpy(),
                                valid_mask_arr=valid_mask.numpy(),
                                return_scale_shift=False,
                            )
                        elif norm_type == "norm":
                            scene_flow_pred_aligned = align_flow_norm_least_square(
                                gt_arr=scene_flow_gt.numpy(),
                                pred_arr=scene_flow_pred.numpy(),
                                valid_mask_arr=valid_mask.numpy(),
                                return_scale_shift=False,
                            )
                        scene_flow_pred = torch.from_numpy(
                            scene_flow_pred_aligned).float()

                        val_metrics[output_type].update(scene_flow_preds=scene_flow_pred.unsqueeze(0),
                                                        scene_flow_gts=scene_flow_gt,
                                                        valid_masks=valid_mask)

                    if output_type in per_task_save_to_dir:
                        img_name = batch["rgb_relative_path"][0].replace(
                            "/", "_")
                        base_name, ext = os.path.splitext(img_name)
                        if mode == "draw_video_results":
                            save_name = f"{base_name}{ext}"
                        else:
                            save_name = f"{base_name}_sceneflow{ext}"
                        save_path = os.path.join(
                            per_task_save_to_dir[output_type], save_name)
                        if rgb_next_int is not None:
                            rgb_next_int = rgb_next_int.squeeze(
                                0).detach().cpu().numpy()
                        rgb_int = rgb_int.squeeze(0).detach().cpu().numpy()

                        visualize_scene_flow_pred_only(
                            flow_pred=scene_flow_pred.detach().cpu().numpy(),
                            png_save_path=save_path)
                        if "scene_flow" in batch:
                            visualize_scene_flow_pred_only(
                                flow_pred=scene_flow_gt.squeeze(
                                    0).detach().cpu().numpy(),
                                png_save_path=save_path.replace(ext, f"_gt{ext}"))

                elif output_type == "depth":
                    depth_pred: np.ndarray = pipe_out.depth_np

                    if "depth_raw_linear" in batch:
                        # GT depth
                        depth_raw_ts = batch["depth_raw_linear"].squeeze()
                        depth_raw = depth_raw_ts.numpy()
                        depth_raw_ts = depth_raw_ts.to(self.accelerator.device)
                        valid_mask_ts = batch["valid_mask_raw"].squeeze()
                        valid_mask = valid_mask_ts.numpy()
                        valid_mask_ts = valid_mask_ts.to(
                            self.accelerator.device)

                        if "least_square" == self.cfg.eval.alignment:
                            depth_pred, scale, shift = align_depth_least_square(
                                gt_arr=depth_raw,
                                pred_arr=depth_pred,
                                valid_mask_arr=valid_mask,
                                return_scale_shift=True,
                                max_resolution=self.cfg.eval.align_max_res,
                            )
                        else:
                            raise RuntimeError(
                                f"Unknown alignment type: {self.cfg.eval.alignment}")

                        # Clip to dataset min max
                        depth_pred = np.clip(
                            depth_pred,
                            a_min=data_loader.dataset.min_depth,
                            a_max=data_loader.dataset.max_depth,
                        )

                        # clip to d > 0 for evaluation
                        depth_pred = np.clip(
                            depth_pred, a_min=1e-6, a_max=None)

                        # Evaluate
                        sample_metric = []
                        depth_pred_ts = torch.from_numpy(
                            depth_pred).to(self.accelerator.device)

                        for met_func in self.depth_metric_funcs:
                            _metric_name = met_func.__name__
                            _metric = met_func(
                                depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
                            sample_metric.append(_metric.__str__())
                            val_metrics[output_type].update(
                                _metric_name, _metric)

                    # Save as 16-bit uint png
                    if output_type in per_task_save_to_dir:
                        img_name = batch["rgb_relative_path"][0].replace(
                            "/", "_")
                        base_name, ext = os.path.splitext(img_name)
                        if mode == "draw_video_results":
                            save_name = f"{base_name}{ext}"
                        else:
                            save_name = f"{base_name}_depth{ext}"
                        save_path = os.path.join(
                            per_task_save_to_dir[output_type], save_name)
                        visualize_depth(depth_pred, save_path)
                        if "depth_raw_linear" in batch:
                            gt_save_path = save_path.replace(ext, f"_gt{ext}")
                            visualize_depth(depth_raw, gt_save_path)
                else:
                    raise ValueError(f"Unknown output type: {output_type}")
        return val_metrics

    def _get_next_seed(self) -> int:
        """Get the next seed from the global seed sequence.

        Returns:
            Next seed value.
        """
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name: str, save_train_state: bool):
        """Save model checkpoint.

        Args:
            ckpt_name: Name of the checkpoint.
            save_train_state: Whether to save training state.
        """
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        unet = self.accelerator.unwrap_model(self.model.unet)
        unet.save_pretrained(unet_path, safe_serialization=False)
        logging.info(f"UNet is saved to: {unet_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_dir: str, load_trainer_state: bool = True, resume_lr_scheduler: bool = True
    ):
        """Load model checkpoint.

        Args:
            ckpt_dir: Directory containing the checkpoint.
            load_trainer_state: Whether to load trainer state.
            resume_lr_scheduler: Whether to resume learning rate scheduler.
        """
        logging.info(f"Loading checkpoint from: {ckpt_dir}")
        # Load UNet

        _model_path = os.path.join(
            ckpt_dir, "unet", "diffusion_pytorch_model.bin")

        self.model.unet.load_state_dict(
            torch.load(_model_path, map_location="cpu")
        )

        logging.info(f"Model parameters are loaded from {ckpt_dir}")

        # Load training states
        if load_trainer_state:

            checkpoint = torch.load(os.path.join(
                ckpt_dir, "trainer.ckpt"), map_location="cpu", weights_only=False)
            self.gradient_accumulation_steps = checkpoint["gradient_accumulation_steps"]
            self.effective_iter = checkpoint["effective_iter"]
            self.iter = self.effective_iter * self.gradient_accumulation_steps
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_dir}")

            if resume_lr_scheduler and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_dir}")

            logging.info(
                f"Checkpoint loaded from: {ckpt_dir}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
            )
        else:
            self.effective_iter = -1
        return

    def _get_backup_ckpt_name(self) -> str:
        """Get backup checkpoint name based on iteration.

        Returns:
            Backup checkpoint name.
        """
        return f"iter_{self.effective_iter:06d}"
