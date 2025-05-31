import os
import tarfile
from io import BytesIO

import numpy as np
import torch
from .semantic import CityscapesEncoder
from PIL import Image
from .base_mtl_dataset import BaseMTLDataset, DepthFileNameMode, DatasetMode


# ['road',
#  'building',
#  'pole',
#  'traffic light',
#  'traffic sign',
#  'vegetation',
#  'sky',
#  'vehicle']

class CityscapesDataset(BaseMTLDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.semantic_encoder = CityscapesEncoder(n_classes=8)
        

    def _read_image(self, img_rel_path) -> np.ndarray:
        image_to_read = os.path.join(self.dataset_dir, img_rel_path)
        image = Image.open(image_to_read)  # [H, W, rgb]
        width, height = image.size
        image = image.resize((width//2, height//2), Image.NEAREST)
        image = np.asarray(image)
        return image

    def _get_data_path(self, index):
        rgb_rel_path = self.filenames[index][0]
        semantic_rel_path = rgb_rel_path.replace("leftImg8bit/", "gtFine/").replace("leftImg8bit", "gtFine_labelIds")
        return rgb_rel_path, semantic_rel_path
    
    
    def _load_semantic_data(self, semantic_rel_path):
        semantic_path = os.path.join(self.dataset_dir, semantic_rel_path)    
        semantic_original_id = Image.open(semantic_path)
        
        width, height = semantic_original_id.size
        semantic_original_id = semantic_original_id.resize((width//2, height//2), Image.NEAREST)
        semantic_original_id = np.asarray(semantic_original_id, dtype=np.uint8)
        semantic_class_id = self.semantic_encoder.encode_segmap(semantic_original_id)
        
        semantic_rgb_int = self.semantic_encoder.segmap2color(semantic_class_id)
        semantic_rgb_norm = semantic_rgb_int / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        semantic_valid_mask = semantic_class_id != self.semantic_encoder.ignore_index
        
        return {
            "semantic_rgb_norm": torch.from_numpy(semantic_rgb_norm).float().permute(2, 0, 1),
            "semantic_class_id": torch.from_numpy(semantic_class_id).long().unsqueeze(0),
            "semantic_valid_mask": torch.from_numpy(semantic_valid_mask).bool().unsqueeze(0)
        }
    
    def _get_data_item(self, index):
        # Special: depth mask is read from data

        rgb_rel_path, semantic_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            
            semantic_data = self._load_semantic_data(semantic_rel_path=semantic_rel_path)
            rasters.update(semantic_data)
        

        other = {"index": index, "rgb_relative_path": rgb_rel_path, "semantic_rel_path": semantic_rel_path}

        return rasters, other
