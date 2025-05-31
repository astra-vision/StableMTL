from .base_mtl_dataset import BaseMTLDataset, DepthFileNameMode, DatasetMode, DatasetConst
import os
import numpy as np
import torch
import pandas as pd
from .augmentation import joint_depth_augmentation, joint_albedo_or_shading_augmentation, joint_normal_augmentation, joint_tasks_augmentation
from torchvision.transforms import InterpolationMode, Resize
from .utils import get_brightness


class MIDIntrinsicDataset(BaseMTLDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs,
        )
        

    def _get_data_path(self, index):
        rgb_rel_path = self.filenames[index][0]
        shading_rel_path = rgb_rel_path.replace(".jpg", "_shading.jpg")
        albedo_rel_path = rgb_rel_path.replace(".jpg", "_albedo.jpg")
        return rgb_rel_path, shading_rel_path, albedo_rel_path

    def _load_shading_data(self, shading_rel_path):
        shading = self._read_image(shading_rel_path).astype(np.float32) / 255.0
        shading = get_brightness(shading)
        shading = torch.from_numpy(shading).float().permute(2, 0, 1)
        return {"shading": shading}

    def _load_albedo_data(self, albedo_rel_path):
        albedo = self._read_image(albedo_rel_path).astype(np.float32) / 255.0
        albedo_valid_mask = self.get_albedo_valid_mask(albedo)
        albedo = torch.from_numpy(albedo).float().permute(2, 0, 1)
        albedo_valid_mask = torch.from_numpy(albedo_valid_mask).float().permute(2, 0, 1)
        return {"albedo": albedo, "albedo_valid_mask": albedo_valid_mask}

    def _get_data_item(self, index):
        # Special: depth mask is read from data

        rgb_rel_path, shading_rel_path, albedo_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            albedo_data = self._load_albedo_data(albedo_rel_path=albedo_rel_path)
            rasters.update(albedo_data)

            shading_data = self._load_shading_data(shading_rel_path=shading_rel_path)
            rasters.update(shading_data)
            shading_valid_mask = albedo_data["albedo_valid_mask"]
            rasters["shading_valid_mask"] = shading_valid_mask

        other = {"index": index, "rgb_relative_path": rgb_rel_path} 
        return rasters, other
    
