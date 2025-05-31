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
Evaluation script for StableMTL: Repurposing Latent Diffusion Models for 
Multi-Task Learning from Partially Annotated Synthetic Datasets.

This script handles the evaluation process for both single-stream and multi-stream models.
It can evaluate models on various tasks including depth, normal maps, semantic segmentation,
optical flow, scene flow, albedo, and shading prediction.

Usage:
    python eval_mtl.py --config config/dataset/dataset_test.yaml --resume_run=/path/to/checkpoint
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
from accelerate.utils import InitProcessGroupKwargs

from src.model.unet import UNet3DConditionModel

from src.stablemtl_pipeline import StableMTLPipeline
from src.dataset import BaseMTLDataset, DatasetMode, get_dataset
from src.dataset.mixed_sampler import MixedBatchSampler
from src.trainer import get_trainer_cls
from src.util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)
from src.util.depth_transform import (
    DepthNormalizerBase,
    get_depth_normalizer,
)
from src.util.logging_util import (
    config_logging,
    init_wandb,
    load_wandb_job_id,
    log_slurm_job_id,
    save_wandb_job_id,
    tb_logger,
)
from diffusers import DDPMScheduler, DDIMScheduler
from src.util.model import setup_unet


def main():
    t_start = datetime.now()
    print(f"start at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_marigold.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--resume_run",
        action="store",
        default=None,
        help="Path of checkpoint to be resumed. If given, will ignore --config, and checkpoint in the config",
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
        "--plot_attn",
        action="store_true",
        default=False,
        help="Plot attention",
    )
    parser.add_argument(
        "--plot_attn_suffix",
        type=str,
        default="",
        help="Suffix for the attention plot",
    )

    parser.add_argument(
        "--infer_budget",
        action="store_true",
        default=False,
        help="Use inference budget",
    )
    parser.add_argument(
        "--infer_budget_task",
        type=str,
        default="",
        help="set of task",
    )
    parser.add_argument(
        "--draw_qualitative_results",
        action="store_true",
        default=False,
        help="Draw qualitative results",
    )
    parser.add_argument(
        "--draw_video_results",
        action="store_true",
        default=False,
        help="Draw video results",
    )


    args = parser.parse_args()




    resume_run = args.resume_run
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
    # Resume previous run
    if resume_run is not None:
        print(f"Resume run: {resume_run}")
        out_dir_run = os.path.dirname(os.path.dirname(resume_run))
        out_tb_dir_run = out_dir_run.replace("output", "output_tb")
        job_name = os.path.basename(out_dir_run)
        # Resume config file
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
        load_cfg = recursive_load_config(args.config)
        if hasattr(load_cfg, "dataset"):
            if hasattr(load_cfg.dataset, "test"):
                cfg.dataset.test = load_cfg.dataset.test
            if hasattr(load_cfg.dataset, "vis"):
                cfg.dataset.vis = load_cfg.dataset.vis
            if hasattr(load_cfg.dataset, "val"):
                cfg.dataset.test = load_cfg.dataset.val

    else:
        # Run from start
        cfg = recursive_load_config(args.config)
        # Full job name
        pure_job_name = os.path.basename(args.config).split(".")[0]
        # Add time prefix
        if args.add_datetime_prefix:
            job_name = f"{t_start.strftime('%y_%m_%d-%H_%M')}-{pure_job_name}"
        else:
            job_name = pure_job_name

        if args.n_gpus > 1:
            job_name = f"{job_name}-gpus{args.n_gpus}"


    cfg_data = cfg.dataset

     # -------------------- Gradient accumulation steps --------------------
    eff_bs = cfg.dataloader.effective_batch_size
    accumulation_steps = eff_bs / (cfg.dataloader.max_train_batch_size * args.n_gpus)
    assert int(accumulation_steps) == accumulation_steps
    accumulation_steps = int(accumulation_steps)

    print(
        f"Effective batch size: {eff_bs}, accumulation steps: {accumulation_steps}"
    )


    kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))

    # -------------------- Accelerator --------------------
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation_steps,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[kwargs]
    )

    out_dir_run = os.path.join("./output", job_name)
    out_tb_dir_run = os.path.join("./output_tb", job_name)
    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")
    out_dir_tb = os.path.join(out_tb_dir_run, "tensorboard")
    out_dir_eval = os.path.join(out_dir_run, "evaluation")
    out_dir_vis = os.path.join(out_dir_run, "visualization")

    latest_ckpt = os.path.join(out_dir_ckpt, "latest")

   


    # Validation dataset
    test_loaders: List[DataLoader] = []

    for _test_dic in cfg_data.test:
        _test_dataset = get_dataset(
            _test_dic,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
        )
        if "debug" in job_name:
            _test_dataset.filenames = _test_dataset.filenames[:5]

        _test_loader = DataLoader(
            dataset=_test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
        )
        test_loaders.append(_test_loader)


    # Visualization dataset
    vis_loaders: List[DataLoader] = []
    if cfg_data.vis is not None:
        for _vis_dic in cfg_data.vis:
            _vis_dataset = get_dataset(
                _vis_dic,
                base_data_dir=base_data_dir,
                mode=DatasetMode.EVAL,
            )
            if "debug" in job_name:
                _vis_dataset.filenames = _vis_dataset.filenames[:5]
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
        os.path.join(base_ckpt_dir, cfg.model.pretrained_path),
        local_files_only=True,
        **_pipeline_kwargs
    )

    model.scheduler = DDIMScheduler.from_config({
        **model.scheduler.config,
        "prediction_type": cfg.model.prediction_type,
    })
    
    setup_unet(model, pretrained_model_path, cfg.trainer.output_types, cfg)



    exclude_mainstream_output_type = True
    if hasattr(cfg.trainer, 'exclude_mainstream_output_type') and cfg.trainer.exclude_mainstream_output_type is not None:
        exclude_mainstream_output_type = cfg.trainer.exclude_mainstream_output_type


    # -------------------- Trainer --------------------
    # Exit time
    if args.exit_after > 0:
        t_end = t_start + timedelta(minutes=args.exit_after)
        logging.info(f"Will exit at {t_end}")
    else:
        t_end = None

    trainer_cls = get_trainer_cls("StableMTLTrainer")
    logging.debug(f"Trainer: {trainer_cls}")

    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataloader=None,
        accelerator=accelerator,
        base_ckpt_dir=base_ckpt_dir,
        out_dir_ckpt=out_dir_ckpt,
        out_dir_eval=out_dir_eval,
        out_dir_vis=out_dir_vis,
        accumulation_steps=accumulation_steps,
        val_dataloaders=test_loaders,
        input_noise=cfg.pipeline.kwargs.input_noise,
        exclude_mainstream_output_type=exclude_mainstream_output_type
    )

    # -------------------- Checkpoint --------------------
    if resume_run is not None and args.plot_attn_suffix != "NoTrain":
        trainer.load_checkpoint(
            resume_run, load_trainer_state=False, resume_lr_scheduler=False
        )
        print(f"Loaded checkpoint from {resume_run}")
    else:
        print(f"No checkpoint loaded")

    trainer.eval()


if "__main__" == __name__:
    main()