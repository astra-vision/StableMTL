import io
import os
import random
import tarfile
from enum import Enum
from typing import Union, List

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize
import torchvision
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time

from src.util.depth_transform import DepthNormalizerBase
from src.util.normal_transform import draw_normal
from src.util.optical_flow_transform import OpticalFlowNormalizerBase

class DatasetMode(Enum):
    RGB_ONLY = "rgb_only"
    EVAL = "evaluate"
    TRAIN = "train"

class DatasetConst:
    OUTPUT_FIELD = "output"
    VALID_MASK_FIELD = "valid_mask"
    OUTPUT_TYPE_FIELD = "output_type"


class DepthFileNameMode(Enum):
    """Prediction file naming modes"""

    id = 1  # id.png
    rgb_id = 2  # rgb_id.png
    i_d_rgb = 3  # i_d_1_rgb.png
    rgb_i_d = 4


def read_image_from_tar(tar_obj, img_rel_path):
    image = tar_obj.extractfile("./" + img_rel_path)
    image = image.read()
    image = Image.open(io.BytesIO(image))


class BaseMTLDataset(Dataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        dataset_dir: str,
        disp_name: str,
        min_depth: float = None,
        max_depth: float = None,
        has_filled_depth: bool = None,
        name_mode: DepthFileNameMode = None,
        output_type: List[str] = None,
        depth_transform: Union[DepthNormalizerBase, None] = None,
        optical_flow_transform: Union[OpticalFlowNormalizerBase, None] = None,
        augmentation_args: dict = None,
        resize_to_hw=None,
        move_invalid_to_far_plane: bool = True,
        rgb_transform=lambda x: x / 255.0 * 2 - 1,  #  [0, 255] -> [-1, 1],
        random_drop_second_frame: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        # dataset info
        self.filename_ls_path = filename_ls_path
        self.dataset_dir = dataset_dir
        self.output_type = output_type

        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset does not exist at: {self.dataset_dir}"
        self.disp_name = disp_name
        self.has_filled_depth = has_filled_depth
        self.name_mode: DepthFileNameMode = name_mode
        self.min_depth = min_depth
        self.max_depth = max_depth

        # training arguments
        self.depth_transform: DepthNormalizerBase = depth_transform
        self.optical_flow_transform: OpticalFlowNormalizerBase = optical_flow_transform
        self.augm_args = augmentation_args
        self.resize_to_hw = resize_to_hw
        # self.random_drop_second_frame = random_drop_second_frame
        self.rgb_transform = rgb_transform
        self.move_invalid_to_far_plane = move_invalid_to_far_plane


        # Load filenames
        with open(self.filename_ls_path, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ]  # [['rgb.png', 'depth.tif'], [], ...]

        
        # Tar dataset
        self.tar_obj = None
        self.is_tar = (
            True
            if os.path.isfile(dataset_dir) and tarfile.is_tarfile(dataset_dir)
            else False
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # start_time = time.perf_counter()
        rasters, other = self._get_data_item(index)
        # if DatasetMode.TRAIN == self.mode:
        #     rasters = self._training_preprocess(rasters, other)
        rasters = self._preprocess(rasters, other)

        # merge
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path, filled_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            depth_data = self._load_depth_data(
                depth_rel_path=depth_rel_path, filled_rel_path=filled_rel_path
            )
            rasters.update(depth_data)


        other = {"index": index, "rgb_relative_path": rgb_rel_path, "depth_rel_path": depth_rel_path}

        return rasters, other

    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

        outputs = {
            "rgb_int": torch.from_numpy(rgb).int(),
            "rgb_norm": torch.from_numpy(rgb_norm).float(),
        }
        return outputs

    def _load_depth_data(self, depth_rel_path, filled_rel_path):
        # Read depth data
        outputs = {}
        depth_raw = self._read_depth_file(depth_rel_path).squeeze()
        depth_raw_linear = torch.from_numpy(depth_raw).float().unsqueeze(0)  # [1, H, W]
        outputs["depth_raw_linear"] = depth_raw_linear.clone()

        if self.has_filled_depth:
            depth_filled = self._read_depth_file(filled_rel_path).squeeze()
            depth_filled_linear = torch.from_numpy(depth_filled).float().unsqueeze(0)
            outputs["depth_filled_linear"] = depth_filled_linear
        else:
            outputs["depth_filled_linear"] = depth_raw_linear.clone()

        return outputs

    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data path
        rgb_rel_path = filename_line[0]

        depth_rel_path, filled_rel_path = None, None
        if DatasetMode.RGB_ONLY != self.mode:
            depth_rel_path = filename_line[1]
            if self.has_filled_depth:
                filled_rel_path = filename_line[2]
        return rgb_rel_path, depth_rel_path, filled_rel_path

    def _read_image(self, img_rel_path) -> np.ndarray:
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            image_to_read = self.tar_obj.extractfile("./" + img_rel_path)
            image_to_read = image_to_read.read()
            image_to_read = io.BytesIO(image_to_read)
        else:
            image_to_read = os.path.join(self.dataset_dir, img_rel_path)
        image = Image.open(image_to_read)  # [H, W, rgb]
        image = np.asarray(image)
        return image


    def _read_rgb_file(self, rel_path) -> np.ndarray:
        rgb = self._read_image(rel_path)
        rgb = np.transpose(rgb, (2, 0, 1)).astype(int)  # [rgb, H, W]
        return rgb


    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        #  Replace code below to decode depth according to dataset definition
        depth_decoded = depth_in

        return depth_decoded


    def _get_valid_mask(self, depth: torch.Tensor):
        # valid_mask = torch.logical_and(
        #     (depth > self.min_depth), (depth < self.max_depth)
        # ).bool()
        valid_mask = np.logical_and(
            (depth > self.min_depth), (depth < self.max_depth)
        )

        return valid_mask


    def get_albedo_valid_mask(self, albedo):
        """
        Create a mask for valid albedo pixels where any RGB value is less than 0.004.

        Args:
            albedo: Albedo image with values in range [0, 255]

        Returns:
            Binary mask where True indicates valid pixels and False indicates invalid pixels
        """
        # Convert to float in range [0, 1] if needed
        if albedo.max() > 1.0:
            albedo_norm = albedo / 255.0
        else:
            albedo_norm = albedo

        # Create mask where any RGB channel is less than 0.004
        invalid_mask = np.any(albedo_norm < 0.004, axis=2)

        # Convert to valid mask (True for valid pixels)
        valid_mask = ~invalid_mask

        # Return as binary mask with same shape as input but single channel
        return valid_mask[..., np.newaxis]


    def _preprocess(self, rasters, other):
        # Augmentation
        # if self.augm_args is not None:
            # rasters = self._augment_data(rasters, other)



        # Normalization
        # rasters["depth_raw_norm"] = self.depth_transform(
        #     rasters["depth_raw_linear"], rasters["valid_mask_raw"]
        # ).clone()
        # rasters["depth_filled_norm"] = self.depth_transform(
        #     rasters["depth_filled_linear"], rasters["valid_mask_filled"]
        # ).clone()
        # rasters["optical_flow_norm"] = self.optical_flow_transform(
        #     rasters["optical_flow"], rasters["optical_flow_valid_mask"]
        # ).clone()

        # Set invalid pixel to far plane
        # if self.move_invalid_to_far_plane:
        #     if self.depth_transform.far_plane_at_max:
        #         rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
        #             self.depth_transform.norm_max
        #         )
        #     else:
        #         rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
        #             self.depth_transform.norm_min
        #         )
        # if self.random_drop_second_frame > 0:
        #     if random.random() < self.random_drop_second_frame:
        #         import pdb; pdb.set_trace()
        # Resize
        if self.resize_to_hw is not None:
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
            )
            rasters = {k: resize_transform(v) for k, v in rasters.items()}

        return rasters



    def _augment_data(self, rasters_dict, other_dict):
        # lr flipping
        lr_flip_p = self.augm_args.lr_flip_p
        output_type = other_dict[DatasetConst.OUTPUT_TYPE_FIELD]
        if random.random() < lr_flip_p:
        # if True:
            # draw_normal(rasters_dict['normal'], "test.png")
            # rasters_dict = {k: v.flip(-1) for k, v in rasters_dict.items()}

            for k, v in rasters_dict.items():
                v = v.flip(-1)
                if k == DatasetConst.OUTPUT_FIELD:
                    if (output_type == "normal" \
                        or output_type == optical_flow \
                        or output_type == "scene_flow"):
                        v[0, :, :] = -v[0, :, :]

                rasters_dict[k] = v

        # if random.random() < self.augm_args.ud_flip_p:
        #     for k, v in rasters_dict.items():
        #         import pdb; pdb.set_trace()

        if self.augm_args.color_jitter.enabled:
            # Stack both images and apply same color jitter transform
            imgs = rasters_dict['rgb_int']
            # import pdb; pdb.set_trace()
            imgs = torch.stack([imgs[:3, :, :], imgs[3:, :, :]], dim=0) # Stack along batch dim

            cj_module = torchvision.transforms.ColorJitter(
                self.augm_args.color_jitter.brightness,
                self.augm_args.color_jitter.contrast,
                self.augm_args.color_jitter.saturation,
                self.augm_args.color_jitter.hue
            )

            imgs = cj_module(imgs / 255.0) * 255.0 # Apply same transform to both
            img1, img2 = imgs[0], imgs[1] # Split back into two images

            rasters_dict['rgb_int'] = torch.cat([img1, img2], dim=0)
            rasters_dict['rgb_norm'] = torch.cat([img1, img2], dim=0) / 255.0 * 2.0 - 1.0
            # Save images to disk
            # img1_pil = torchvision.transforms.ToPILImage()(img1.byte())
            # img2_pil = torchvision.transforms.ToPILImage()(img2.byte())
            # img1_pil.save('img1.png')
            # img2_pil.save('img2.png')
            # import pdb; pdb.set_trace()

        return rasters_dict

    def __del__(self):
        if hasattr(self, "tar_obj") and self.tar_obj is not None:
            self.tar_obj.close()
            self.tar_obj = None


def get_pred_name(rgb_basename, name_mode, suffix=".png"):
    if DepthFileNameMode.rgb_id == name_mode:
        pred_basename = "pred_" + rgb_basename.split("_")[1]
    elif DepthFileNameMode.i_d_rgb == name_mode:
        pred_basename = rgb_basename.replace("_rgb.", "_pred.")
    elif DepthFileNameMode.id == name_mode:
        pred_basename = "pred_" + rgb_basename
    elif DepthFileNameMode.rgb_i_d == name_mode:
        pred_basename = "pred_" + "_".join(rgb_basename.split("_")[1:])
    else:
        raise NotImplementedError
    # change suffix
    pred_basename = os.path.splitext(pred_basename)[0] + suffix

    return pred_basename
