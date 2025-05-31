import os
import tarfile
from io import BytesIO

import numpy as np
import torch

from .base_mtl_dataset import BaseMTLDataset, DepthFileNameMode, DatasetMode


class DIODEDataset(BaseMTLDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # DIODE data parameter
            min_depth=0.6,
            max_depth=350,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )


    def _read_npy_file(self, rel_path):
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            fileobj = self.tar_obj.extractfile("./" + rel_path)
            npy_path_or_content = BytesIO(fileobj.read())
        else:
            npy_path_or_content = os.path.join(self.dataset_dir, rel_path)
        data = np.load(npy_path_or_content).squeeze()[np.newaxis, :, :]
        return data

    def _read_depth_file(self, rel_path):
        depth = self._read_npy_file(rel_path)
        return depth

    def _get_data_path(self, index):
        rgb_rel_path, depth_rel_path, mask_rel_path = self.filenames[index]
        normal_rel_path = depth_rel_path.replace("_depth", "_normal")
        normal_rel_path = os.path.join(self.dataset_dir, normal_rel_path)
        return rgb_rel_path, depth_rel_path, mask_rel_path, normal_rel_path

    def _load_normal_data(self, normal_rel_path):
        normal_data = self._read_npy_file(normal_rel_path).squeeze()
        # import matplotlib.pyplot as plt
        # plt.imsave("normal_diode_before.png", (normal_data+1)/2)

        normal_data[:, :, 2] = -normal_data[:, :, 2] # flip z axis
        normal_data[:,:, 1] = -normal_data[:,:, 1] # flip y axis
        normal_data = -normal_data # invert normal to get outward normal

        # plt.imsave("normal_diode.png", (normal_data+1)/2)
        # import pdb;pdb.set_trace()
        normal_data = torch.from_numpy(normal_data).float().permute(2, 0, 1)
        normal_norm = torch.norm(normal_data, dim=0, keepdim=True)
        normal_valid_mask = (normal_norm > 0.5) & (normal_norm < 1.5)
        return {"normal": normal_data, "normal_valid_mask": normal_valid_mask}

    def _get_data_item(self, index):
        # Special: depth mask is read from data

        rgb_rel_path, depth_rel_path, mask_rel_path, normal_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            depth_data = self._load_depth_data(
                depth_rel_path=depth_rel_path, filled_rel_path=None
            )
            rasters.update(depth_data)

            # valid mask
            mask = self._read_npy_file(mask_rel_path).astype(bool)
            mask = torch.from_numpy(mask).bool()
            rasters["valid_mask_raw"] = mask.clone()
            rasters["valid_mask_filled"] = mask.clone()


            normal_data = self._load_normal_data(normal_rel_path=normal_rel_path)
            rasters.update(normal_data)

        other = {"index": index, "rgb_relative_path": rgb_rel_path}
        # full size: 768 x 1024
        
        return rasters, other
