from .base_mtl_dataset import BaseMTLDataset, DepthFileNameMode, DatasetMode, DatasetConst
import os
import numpy as np
import torch
from .augmentation import joint_depth_augmentation, joint_albedo_or_shading_augmentation, joint_normal_augmentation, joint_tasks_augmentation
from torchvision.transforms import InterpolationMode, Resize
from .utils import get_brightness


class HypersimDataset(BaseMTLDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # Hypersim data parameter
            min_depth=1e-5,
            max_depth=65.0,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.rgb_i_d,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode Hypersim depth
        depth_decoded = depth_in / 1000.0
        return depth_decoded

    def _load_depth_data(self, depth_rel_path, filled_rel_path):
        depth_data = super()._load_depth_data(depth_rel_path, filled_rel_path)
        return depth_data

    def _load_normal_data(self, normal_npy_rel_path, normal_valid_mask_path):
        normal_npy_path = os.path.join(self.dataset_dir, normal_npy_rel_path)
        normal_valid_mask_path = os.path.join(self.dataset_dir, normal_valid_mask_path)
        normal = np.load(normal_npy_path)
        # normal = np.transpose(normal, (2, 0, 1))
        valid_mask = np.load(normal_valid_mask_path)
        valid_mask = valid_mask[..., np.newaxis]
        # valid_mask = valid_mask[np.newaxis, :, :]

        # normal = torch.from_numpy(normal).float()
        # valid_mask = torch.from_numpy(valid_mask).bool()

        return normal, valid_mask




    def _get_data_path(self, index):
        raise NotImplementedError("This method should be implemented in the subclass.")


class HypersimAlbedoDataset(HypersimDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs,
        )



    def _get_data_item(self, index):
        rgb_rel_path, albedo_rel_path = self._get_data_path(index=index)

        img0 = self._read_image(rgb_rel_path) # (H, W, 3)
        albedo = self._read_image(albedo_rel_path) # (H, W, 3)
        albedo_valid_mask = self.get_albedo_valid_mask(albedo)

        if self.mode == DatasetMode.TRAIN:
            img0, albedo, albedo_valid_mask = joint_albedo_or_shading_augmentation(
                img0, albedo, albedo_valid_mask, self.augm_args['hypersim_albedo']
            )

        img0 = img0.transpose(2, 0, 1)
        albedo = albedo.transpose(2, 0, 1)
        albedo_valid_mask = albedo_valid_mask.transpose(2, 0, 1)

        img0 = torch.from_numpy(img0.copy()).float()
        albedo = torch.from_numpy(albedo.copy()).float()
        albedo_valid_mask = torch.from_numpy(albedo_valid_mask.copy()).bool()

        rasters = {}
        rasters['rgb_int'] = img0
        rasters['rgb_norm'] = img0 / 255.0 * 2.0 - 1.0


        other = {"index": index, "rgb_relative_path": rgb_rel_path,
                    DatasetConst.OUTPUT_TYPE_FIELD: "albedo"}

        if self.mode == DatasetMode.TRAIN:
            rasters[DatasetConst.OUTPUT_FIELD] = albedo / 255.0 * 2.0 - 1.0
            rasters[DatasetConst.VALID_MASK_FIELD] = albedo_valid_mask

        else:
            rasters["albedo"] = albedo / 255.0
            rasters["albedo_valid_mask"] = albedo_valid_mask
            if self.resize_to_hw is not None:
                resize_transform = Resize(
                    size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
                )
                rasters = {k: resize_transform(v) for k, v in rasters.items()}

        return rasters, other


    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data paths
        rgb_rel_path, depth_rel_path, normal_npy_rel_path, _, normal_valid_mask_path  = filename_line
        albedo_rel_path = rgb_rel_path.replace("rgb", "reflectance")
        return rgb_rel_path, albedo_rel_path


class HypersimShadingDataset(HypersimDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs,
        )


    def _get_data_item(self, index):
        rgb_rel_path, shading_rel_path, albedo_rel_path = self._get_data_path(index=index)

        img0 = self._read_image(rgb_rel_path) # (H, W, 3)
        shading = self._read_image(shading_rel_path) # (H, W, 3)
        shading = get_brightness(shading)
        albedo = self._read_image(albedo_rel_path) # (H, W, 3)
        shading_valid_mask = self.get_albedo_valid_mask(albedo)


        if self.mode == DatasetMode.TRAIN:
            img0, shading, shading_valid_mask = joint_albedo_or_shading_augmentation(
                img0, shading, shading_valid_mask, self.augm_args['hypersim_shading']
            )

        img0 = img0.transpose(2, 0, 1)
        shading = shading.transpose(2, 0, 1)
        shading_valid_mask = shading_valid_mask.transpose(2, 0, 1)

        img0 = torch.from_numpy(img0.copy()).float()
        shading = torch.from_numpy(shading.copy()).float()
        shading_valid_mask = torch.from_numpy(shading_valid_mask.copy()).bool()

        rasters = {}
        rasters['rgb_int'] = img0
        rasters['rgb_norm'] = img0 / 255.0 * 2.0 - 1.0

        other = {"index": index, "rgb_relative_path": rgb_rel_path,
                DatasetConst.OUTPUT_TYPE_FIELD: "shading"}

        if self.mode == DatasetMode.TRAIN:
            rasters[DatasetConst.OUTPUT_FIELD] = shading / 255.0 * 2.0 - 1.0
            rasters[DatasetConst.VALID_MASK_FIELD] = shading_valid_mask
        else:
            rasters["shading"] = shading / 255.0
            rasters["shading_valid_mask"] = shading_valid_mask
            if self.resize_to_hw is not None:
                resize_transform = Resize(
                    size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
                )
                rasters = {k: resize_transform(v) for k, v in rasters.items()}

        return rasters, other


    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data paths
        rgb_rel_path, depth_rel_path, normal_npy_rel_path, _, normal_valid_mask_path  = filename_line
        shading_rel_path = rgb_rel_path.replace("rgb", "shading")
        albedo_rel_path = rgb_rel_path.replace("rgb", "reflectance")
        return rgb_rel_path, shading_rel_path, albedo_rel_path


class HypersimDepthDataset(HypersimDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs,
        )


    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path, normal_npy_rel_path, normal_valid_mask_path = self._get_data_path(index=index)

        img0 = self._read_image(rgb_rel_path) # (H, W, 3)
        depth_raw = self._read_depth_file(depth_rel_path)[..., np.newaxis] # (H, W, 1)
        valid_mask = self._get_valid_mask(depth_raw) # (H, W, 1)

        img0, depth_raw, valid_mask = joint_depth_augmentation(
            img0, depth_raw, valid_mask, self.augm_args['hypersim_depth']
        )

        img0 = img0.transpose(2, 0, 1)
        depth_raw = depth_raw.transpose(2, 0, 1)
        valid_mask = valid_mask.transpose(2, 0, 1)

        img0 = torch.from_numpy(img0.copy()).float()
        depth_raw = torch.from_numpy(depth_raw.copy()).float()
        valid_mask = torch.from_numpy(valid_mask.copy()).bool()

        rasters = {}
        rasters['rgb_int'] = img0
        rasters['rgb_norm'] = img0 / 255.0 * 2.0 - 1.0


        # depth_map = depth_map.repeat(3, 1, 1)
        if DatasetMode.TRAIN == self.mode:
            depth_map_norm = self.depth_transform(depth_raw, valid_mask)

        rasters[DatasetConst.OUTPUT_FIELD] = depth_map_norm
        rasters[DatasetConst.VALID_MASK_FIELD] = valid_mask
        other = {"index": index, "rgb_relative_path": rgb_rel_path,
                 DatasetConst.OUTPUT_TYPE_FIELD: "depth"}

        return rasters, other


    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data paths
        rgb_rel_path, depth_rel_path, normal_npy_rel_path, _, normal_valid_mask_path  = filename_line

        # depth_rel_path, filled_rel_path = None, None
        # if DatasetMode.RGB_ONLY != self.mode:
        #     depth_rel_path = filename_line[1]
            # if self.has_filled_depth:
            #     filled_rel_path = filename_line[2]
        # return rgb_rel_path, depth_rel_path, filled_rel_path
        # return rgb_rel_path, depth_rel_path
        return rgb_rel_path, depth_rel_path, normal_npy_rel_path, normal_valid_mask_path


class HypersimNormalDataset(HypersimDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs,
        )

    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path, normal_npy_rel_path, normal_valid_mask_path = self._get_data_path(index=index)

        img0 = self._read_image(rgb_rel_path) # (H, W, 3)
        normal, valid_mask = self._load_normal_data(normal_npy_rel_path, normal_valid_mask_path) # (H, W, 3)

        img0, normal, valid_mask = joint_normal_augmentation(
            img0, normal, valid_mask, self.augm_args['hypersim_normal']
        )

        img0 = img0.transpose(2, 0, 1)
        normal = normal.transpose(2, 0, 1)
        valid_mask = valid_mask.transpose(2, 0, 1)

        img0 = torch.from_numpy(img0.copy()).float()
        normal = torch.from_numpy(normal.copy()).float()
        valid_mask = torch.from_numpy(valid_mask.copy()).bool()

        rasters = {}
        rasters['rgb_int'] = img0
        rasters['rgb_norm'] = img0 / 255.0 * 2.0 - 1.0


        rasters[DatasetConst.OUTPUT_FIELD] = normal
        rasters[DatasetConst.VALID_MASK_FIELD] = valid_mask
        other = {"index": index, "rgb_relative_path": rgb_rel_path,
                 DatasetConst.OUTPUT_TYPE_FIELD: "normal"}

        return rasters, other


    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data paths
        rgb_rel_path, depth_rel_path, normal_npy_rel_path, _, normal_valid_mask_path  = filename_line

        return rgb_rel_path, depth_rel_path, normal_npy_rel_path, normal_valid_mask_path


