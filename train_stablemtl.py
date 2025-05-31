#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2024 StableMTL Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Main training script for StableMTL: Repurposing Latent Diffusion Models for 
Multi-Task Learning from Partially Annotated Synthetic Datasets.

This script handles the training process for both single-stream and multi-stream models.
It sets up the datasets, model configuration, and training loop based on command line arguments
and configuration files.

Usage:
    python train_stablemtl.py --config config/train_stablemtl.yaml [options]
"""

import argparse
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import List

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, DistributedDataParallelKwargs

from src.stablemtl_pipeline import StableMTLPipeline
from src.dataset import BaseMTLDataset, DatasetMode, get_dataset
from src.dataset.mixed_sampler import MixedBatchSampler
from src.trainer import get_trainer_cls
from src.util.config_util import recursive_load_config
from src.util.depth_transform import (
    DepthNormalizerBase,
    get_depth_normalizer,
)
from src.util.optical_flow_transform import (
    OpticalFlowNormalizerBase,
    get_optical_flow_normalizer,
)
from src.util.logging_util import (
    config_logging,
    log_slurm_job_id,
    tb_logger,
)

from diffusers import DDPMScheduler, DDIMScheduler
from src.model.unet import UNet3DConditionModel
from src.util.model import setup_unet


def main():
    t_start = datetime.now()
    print(f"start at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_debug.yaml",
        help="Path to config file.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Do not use cuda.")
    parser.add_argument(
        "--exit_after",
        type=int,
        default=-1,
        help="Save checkpoint and exit after X minutes.",
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=1,
        help="number of gpus",
    )
    parser.add_argument("--no_wandb", action="store_true", help="run without wandb")
    parser.add_argument(
        "--do_not_copy_data",
        action="store_true",
        help="On Slurm cluster, do not copy data to local scratch",
    )
    parser.add_argument(
        "--base_data_dir", type=str, default=None, help="directory of training data"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="directory to save output"
    )
    parser.add_argument(
        "--base_ckpt_dir",
        type=str,
        default=None,
        help="directory of pretrained checkpoint",
    )
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_true",
        help="Add datetime to the output folder name",
    )
    parser.add_argument(
        "--subfix",
        default="",
        help="Subfix to the output folder name",
    )

    parser.add_argument(
        "--no_lr_scheduler",
        action="store_true",
        default=False,
        help="No lr scheduler",
    )


    args = parser.parse_args()




    output_dir = args.output_dir
    base_data_dir = (
        args.base_data_dir
        if args.base_data_dir is not None
        else os.environ["BASE_DATA_DIR"]
    )
    base_ckpt_dir = (
        args.base_ckpt_dir
        if args.base_ckpt_dir is not None
        else os.environ["BASE_CKPT_DIR"]
    )



    # -------------------- Initialization --------------------
    cfg = recursive_load_config(args.config)
    # Full job name
    pure_job_name = os.path.basename(args.config).split(".")[0]
    # Add time prefix
    if args.add_datetime_prefix:
        job_name = f"{t_start.strftime('%y_%m_%d-%H_%M')}-{pure_job_name}"
    else:
        job_name = pure_job_name

    if args.subfix:
        job_name = f"{job_name}-{args.subfix}"


    cfg_data = cfg.dataset

     # -------------------- Gradient accumulation steps --------------------
    eff_bs = cfg.dataloader.effective_batch_size
    accumulation_steps = eff_bs / (cfg.dataloader.max_train_batch_size * args.n_gpus)
    assert int(accumulation_steps) == accumulation_steps
    accumulation_steps = int(accumulation_steps)

    print(
        f"Effective batch size: {eff_bs}, accumulation steps: {accumulation_steps}"
    )


    # kwargs_ddp = DistributedDataParallelKwargs(find_unused_parameters=False)
    kwargs_ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=3))

    # -------------------- Accelerator --------------------
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation_steps,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[kwargs, kwargs_ddp],
    )


    out_dir_run = os.path.join(output_dir, job_name)
    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")
    out_dir_tb = os.path.join(out_dir_run, "tensorboard")
    out_dir_eval = os.path.join(out_dir_run, "evaluation")
    out_dir_vis = os.path.join(out_dir_run, "visualization")

    latest_ckpt = os.path.join(out_dir_ckpt, "latest")
    
    
    if not os.path.exists(latest_ckpt):
        if accelerator.is_main_process:
            os.makedirs(out_dir_run, exist_ok=True)
            # Other directories
            os.makedirs(out_dir_ckpt, exist_ok=True)

            os.makedirs(out_dir_tb, exist_ok=True)

            os.makedirs(out_dir_eval, exist_ok=True)

            os.makedirs(out_dir_vis, exist_ok=True)
        resume_run = None
    else:
        resume_run = os.path.join(out_dir_ckpt, "latest")

    # -------------------- Logging settings --------------------
    config_logging(cfg.logging, out_dir=out_dir_run)
    logging.debug(f"config: {cfg}")



    # Tensorboard (should be initialized after wandb)
    tb_logger.set_dir(out_dir_tb)

    log_slurm_job_id(step=0)

    # -------------------- Snapshot of code and config --------------------
    if resume_run is None:
        _output_path = os.path.join(out_dir_run, "config.yaml")
        with open(_output_path, "w+") as f:
            OmegaConf.save(config=cfg, f=f)
        logging.info(f"Config saved to {_output_path}")
        # Copy and tar code on the first run
        _temp_code_dir = os.path.join(out_dir_run, "code_tar")
        _code_snapshot_path = os.path.join(out_dir_run, "code_snapshot.tar")
        os.system(
            f"rsync --relative -arhvz --quiet --filter=':- .gitignore' --exclude '.git' . '{_temp_code_dir}'"
        )
        os.system(f"tar -cf {_code_snapshot_path} {_temp_code_dir}")
        os.system(f"rm -rf {_temp_code_dir}")
        logging.info(f"Code snapshot saved to: {_code_snapshot_path}")





    # -------------------- Data --------------------
    loader_seed = cfg.dataloader.seed
    if loader_seed is None:
        loader_generator = None
    else:
        # Add process rank to seed for better randomness across processes
        process_rank = accelerator.process_index
        loader_generator = torch.Generator().manual_seed(loader_seed + process_rank)


    # Training dataset
    depth_transform: DepthNormalizerBase = get_depth_normalizer(
        cfg_normalizer=cfg.depth_normalization
    )
    optical_flow_transform: OpticalFlowNormalizerBase = get_optical_flow_normalizer(
        cfg_normalizer=cfg.optical_flow_normalization
    )
    train_dataset: BaseMTLDataset = get_dataset(
        cfg_data.train,
        base_data_dir=base_data_dir,
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
        depth_transform=depth_transform,
        optical_flow_transform=optical_flow_transform,
    )
    logging.debug("Augmentation: ", cfg.augmentation)
    print("cfg.trainer.prob_drop_second_frame: ", cfg.trainer.prob_drop_second_frame)
    if "mixed" == cfg_data.train.name:
        total_prob = sum(cfg_data.train.prob_ls)
        cfg_data.train.prob_ls = [p / total_prob for p in cfg_data.train.prob_ls]
        print(">>>> cfg_data.train.prob_ls: ", cfg_data.train.prob_ls)
        assert len(cfg_data.train.prob_ls) == len(
            train_dataset
        ), "Lengths don't match: `prob_ls` and `dataset_list`"
        dataset_ls = [
            dataset for dataset in train_dataset
            if dataset.output_type in cfg.trainer.output_types
        ]
        print(">>>> dataset_ls: ", dataset_ls)
       

        concat_dataset = ConcatDataset(dataset_ls)
        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=cfg.dataloader.max_train_batch_size,
            drop_last=True,
            accumulation_steps=accumulation_steps,
            prob=cfg_data.train.prob_ls,
            shuffle=True,
            iterative_sampling=cfg.dataloader.iterative_sampling,
            generator=loader_generator,
        )
        train_loader = DataLoader(
            concat_dataset,
            batch_sampler=mixed_sampler,
            num_workers=cfg.dataloader.num_workers,
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.dataloader.max_train_batch_size,
            num_workers=cfg.dataloader.num_workers,
            shuffle=True,
            generator=loader_generator,
        )

    # Validation dataset
    val_loaders: List[DataLoader] = []

    for _val_dic in cfg_data.val:
        _val_dataset = get_dataset(
            _val_dic,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
        )

        if not len(set(_val_dataset.output_type) & set(cfg.trainer.output_types)) > 0:
            continue

        _val_loader = DataLoader(
            dataset=_val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
        )
        val_loaders.append(_val_loader)


    # Visualization dataset
    vis_loaders: List[DataLoader] = []
    if cfg_data.vis is not None:
        for _vis_dic in cfg_data.vis:
            _vis_dataset = get_dataset(
                _vis_dic,
                base_data_dir=base_data_dir,
                mode=DatasetMode.EVAL,
            )
            if not len(set(_vis_dataset.output_type) & set(cfg.trainer.output_types)) > 0:
                continue

            _vis_loader = DataLoader(
                dataset=_vis_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfg.dataloader.num_workers,
            )
            vis_loaders.append(_vis_loader)

    # -------------------- Model --------------------
    _pipeline_kwargs = cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}
    pretrained_model_path = os.path.join(base_ckpt_dir, cfg.model.pretrained_path)
    model = StableMTLPipeline.from_pretrained(
        pretrained_model_path,
        local_files_only=True,
        **_pipeline_kwargs
    )

    model.scheduler = DDIMScheduler.from_config({
        **model.scheduler.config,
        "prediction_type": cfg.model.prediction_type,
    })

    
    # Call setup_unet with simplified parameters - all config values are extracted inside the function
    setup_unet(model, pretrained_model_path, cfg.trainer.output_types, cfg)

  



    # -------------------- Trainer --------------------
    # Exit time
    if args.exit_after > 0:
        t_end = t_start + timedelta(minutes=args.exit_after)
        logging.info(f"Will exit at {t_end}")
    else:
        t_end = None

    trainer_cls = get_trainer_cls(cfg.trainer.name)
    logging.debug(f"Trainer: {trainer_cls}")
    
    exclude_mainstream_output_type = True
    if hasattr(cfg.trainer, 'exclude_mainstream_output_type') and cfg.trainer.exclude_mainstream_output_type is not None:
        exclude_mainstream_output_type = cfg.trainer.exclude_mainstream_output_type
        
    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataloader=train_loader,
        accelerator=accelerator,
        base_ckpt_dir=base_ckpt_dir,
        out_dir_ckpt=out_dir_ckpt,
        out_dir_eval=out_dir_eval,
        out_dir_vis=out_dir_vis,
        accumulation_steps=accumulation_steps,
        val_dataloaders=val_loaders,
        vis_dataloaders=vis_loaders,
        input_noise=cfg.pipeline.kwargs.input_noise,
        no_lr_scheduler=args.no_lr_scheduler,
        exclude_mainstream_output_type=exclude_mainstream_output_type,
    )

    # -------------------- Checkpoint --------------------
    if resume_run is not None:
        trainer.load_checkpoint(
            resume_run, load_trainer_state=True, resume_lr_scheduler=not args.no_lr_scheduler
        )
    # -------------------- Training & Evaluation Loop --------------------
    # try:
    # Get optional unet_child component
    trainer.model.unet_child = getattr(trainer.model, "unet_child", None)

    # Prepare model components, optimizer, dataloader and scheduler for distributed training
    components_to_prepare = [
        trainer.model.unet,
        trainer.optimizer,
        trainer.lr_scheduler,
    ]
    trainer.optimizer.zero_grad()

    # Add unet_child to preparation if it exists
    if trainer.model.unet_child is not None:
        components_to_prepare.insert(1, trainer.model.unet_child)
        prepared_components = trainer.accelerator.prepare(*components_to_prepare)
        trainer.model.unet, trainer.model.unet_child, trainer.optimizer, trainer.lr_scheduler = prepared_components
    else:
        prepared_components = trainer.accelerator.prepare(*components_to_prepare)
        trainer.model.unet, trainer.optimizer, trainer.lr_scheduler = prepared_components

    trainer.train(t_end=t_end)
  

if "__main__" == __name__:
    main()