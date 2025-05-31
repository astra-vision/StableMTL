import torch
import os
import numpy as np

from .base_mtl_dataset import BaseMTLDataset, DepthFileNameMode, DatasetMode, DatasetConst
from .kitti_dataset import KITTIDataset
from PIL import Image
from .semantic import VKitti2Encoder
import cv2
from .utils import kitti_benchmark_crop
from src.util.optical_flow_transform  import ScaleShiftOpticalFlowNormalizer
import torch.nn.functional as F
from .flow_augmentation import joint_optical_flow_augmentation, joint_scene_flow_augmentation, joint_flows_augmentation
from src.util.alignment import normalize_optical_flow, normalize_scene_flow
from .augmentation import joint_tasks_augmentation


class FlyingThings3DDataset(BaseMTLDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )
        self.crop_size = [960, 536]
        self.f, self.cx, self.cy = 1050.0, 479.5, 269.5

    # ====== stablepoop ======
    def _get_data_item(self, index):
        raise NotImplementedError("Please implement this method in the derived class.")





class FlyingThings3DOpticalFlowDataset(FlyingThings3DDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_type = "optical_flow"

    @staticmethod
    def load_flow_png(filepath, scale=64.0):
        # for KITTI which uses 16bit PNG images
        # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
        # The -1 is here to specify not to change the image depth (16bit), and is compatible
        # with both OpenCV2 and OpenCV3
        flow_img = cv2.imread(filepath, -1)
        flow = flow_img[:, :, 2:0:-1].astype(np.float32)
        mask = flow_img[:, :, 0] > 0
        flow = flow - 32768.0
        flow = flow / scale
        return flow, mask

    def crop(self, image):
        crop_w, crop_h = self.crop_size
        start_w = (image.shape[1] - crop_w) // 2
        start_h = (image.shape[0] - crop_h) // 2
        return image[start_h:start_h+crop_h, start_w:start_w+crop_w]

    def _get_data_item(self, index):
        rgb_rel_path, next_rgb_rel_path, optical_flow_rel_path = self._get_data_path(index=index)

        rasters = {}

        optical_flow, optical_flow_valid_mask = self.load_flow_png(os.path.join(self.dataset_dir, optical_flow_rel_path))
        optical_flow_valid_mask = optical_flow_valid_mask[:, :, np.newaxis]
        img0 = self._read_image(rgb_rel_path)
        img1 = self._read_image(next_rgb_rel_path)
        img0 = self.crop(img0)
        img1 = self.crop(img1)
        optical_flow = self.crop(optical_flow)
        optical_flow_valid_mask = self.crop(optical_flow_valid_mask)

        if DatasetMode.TRAIN == self.mode:
            img0, img1, optical_flow, optical_flow_valid_mask = joint_optical_flow_augmentation(
                img0, img1, optical_flow, optical_flow_valid_mask, self.augm_args['flyingthings3d_optical_flow']
            )

        img0 = img0.transpose(2, 0, 1)
        img1 = img1.transpose(2, 0, 1)
        optical_flow = optical_flow.transpose(2, 0, 1)
        optical_flow_valid_mask = optical_flow_valid_mask.transpose(2, 0, 1)

        img0 = torch.from_numpy(img0.copy()).float()
        img1 = torch.from_numpy(img1.copy()).float()
        optical_flow = torch.from_numpy(optical_flow.copy()).float()
        optical_flow_valid_mask = torch.from_numpy(optical_flow_valid_mask.copy()).bool()


        rasters['rgb_int'] = img0
        rasters['rgb_norm'] = img0 / 255.0 * 2.0 - 1.0

        rasters['rgb_next_int'] = img1
        rasters['rgb_next_norm'] = img1 / 255.0 * 2.0 - 1.0

        # Get max values
        max_x = max(abs(optical_flow[0, :, :].max()), abs(optical_flow[0, :, :].min()))
        max_y = max(abs(optical_flow[1, :, :].max()), abs(optical_flow[1, :, :].min()))
    

        # Only normalize if there is motion
        norm_type = "hw"
        # norm_type = "norm"
        optical_flow_norm = normalize_optical_flow(optical_flow, type=norm_type)



        rasters[DatasetConst.OUTPUT_FIELD] = optical_flow_norm
        rasters[DatasetConst.VALID_MASK_FIELD] = optical_flow_valid_mask
        rasters['optical_flow_raw'] = optical_flow

        other = {"index": index, "rgb_relative_path": rgb_rel_path,
                 DatasetConst.OUTPUT_TYPE_FIELD: "optical_flow"}
        return rasters, other

    def _get_data_path(self, index):
        filename_line = self.filenames[index] # Scene02/clone/frames/depth/Camera_0/depth_00000.png
        # Get data path
        optical_flow_rel_path = filename_line[0]
        rgb_rel_path = optical_flow_rel_path.replace("flow_2d", "image_clean")

        frame_id = int(rgb_rel_path.split('/')[-1].split('_')[-1].replace('.png', ''))
        next_frame_id = frame_id + 1
        next_rgb_rel_path = rgb_rel_path.replace(f'{frame_id:07d}', f'{next_frame_id:07d}')

        return rgb_rel_path, next_rgb_rel_path, optical_flow_rel_path




class FlyingThings3DSceneFlowDataset(FlyingThings3DDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_type = "scene_flow"

    @staticmethod
    def load_flow_png(filepath, scale=64.0):
        # for KITTI which uses 16bit PNG images
        # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
        # The -1 is here to specify not to change the image depth (16bit), and is compatible
        # with both OpenCV2 and OpenCV3
        flow_img = cv2.imread(filepath, -1)
        flow = flow_img[:, :, 2:0:-1].astype(np.float32)
        mask = flow_img[:, :, 0] > 0
        flow = flow - 32768.0
        flow = flow / scale
        return flow, mask

    def crop(self, image):
        crop_w, crop_h = self.crop_size
        start_w = (image.shape[1] - crop_w) // 2
        start_h = (image.shape[0] - crop_h) // 2
        return image[start_h:start_h+crop_h, start_w:start_w+crop_w]

    def project_flow_3d_to_2d(self, flow_3d, pc, image_h, image_w, f, cx, cy):
        # flow_3d: (H, W, 3)
        # pc1: (H, W, 3)
        # return: (H, W, 2)
        pc_x, pc_y, depth = pc[..., 0], pc[..., 1], pc[..., 2]

        image_x = cx + (f / depth) * pc_x
        image_y = cy + (f / depth) * pc_y

        image_x = np.round(image_x).astype(np.int32)
        image_y = np.round(image_y).astype(np.int32)

        # Ensure the projected pixels are within the image boundaries
        valid_mask = (image_x >= 0) & (image_x < image_w) & (image_y >= 0) & (image_y < image_h)
        image_x = image_x[valid_mask]
        image_y = image_y[valid_mask]
        flow_3d = flow_3d[valid_mask]

        valid_mask = np.zeros((image_h, image_w, 1), dtype=bool)
        valid_mask[image_y, image_x, 0] = True

        scene_flow = np.zeros((image_h, image_w, 3))
        scene_flow[image_y, image_x, :] = flow_3d

        return scene_flow, valid_mask


    def _get_data_item(self, index):
        rgb_rel_path, next_rgb_rel_path, optical_flow_rel_path, pc_rel_path, flow_3d_rel_path = self._get_data_path(index=index)

        rasters = {}

        optical_flow, optical_flow_valid_mask = self.load_flow_png(os.path.join(self.dataset_dir, optical_flow_rel_path))
        # optical_flow_valid_mask = optical_flow_valid_mask[:, :, np.newaxis]
        img0 = self._read_image(rgb_rel_path)
        img1 = self._read_image(next_rgb_rel_path)
        img0 = self.crop(img0)
        img1 = self.crop(img1)

        pc_dict = np.load(os.path.join(self.dataset_dir, pc_rel_path))
        flow_3d = np.load(os.path.join(self.dataset_dir, flow_3d_rel_path))
        pc1 = pc_dict['pc1']

        image_h, image_w = img0.shape[:2]
        scene_flow, scene_flow_valid_mask = self.project_flow_3d_to_2d(flow_3d, pc1, image_h, image_w, self.f, self.cx, self.cy)

        if DatasetMode.TRAIN == self.mode:
            img0, img1, scene_flow, scene_flow_valid_mask = joint_scene_flow_augmentation(
                img0, img1, scene_flow, scene_flow_valid_mask, self.augm_args['flyingthings3d_scene_flow']
            )

        img0 = img0.transpose(2, 0, 1)
        img1 = img1.transpose(2, 0, 1)
        scene_flow = scene_flow.transpose(2, 0, 1)
        scene_flow_valid_mask = scene_flow_valid_mask.transpose(2, 0, 1)

        img0 = torch.from_numpy(img0.copy()).float()
        img1 = torch.from_numpy(img1.copy()).float()
        scene_flow = torch.from_numpy(scene_flow.copy()).float()
        scene_flow_valid_mask = torch.from_numpy(scene_flow_valid_mask.copy()).bool()


        rasters['rgb_int'] = img0
        rasters['rgb_norm'] = img0 / 255.0 * 2.0 - 1.0

        rasters['rgb_next_int'] = img1
        rasters['rgb_next_norm'] = img1 / 255.0 * 2.0 - 1.0

        norm_type = "hw"
        # norm_type = "norm"
        scene_flow_norm = normalize_scene_flow(scene_flow, type=norm_type)

        rasters[DatasetConst.OUTPUT_FIELD] = scene_flow_norm
        rasters[DatasetConst.VALID_MASK_FIELD] = scene_flow_valid_mask
        rasters['scene_flow'] = scene_flow

        other = {"index": index, "rgb_relative_path": rgb_rel_path,
                 DatasetConst.OUTPUT_TYPE_FIELD: "scene_flow"}
        return rasters, other

    def _get_data_path(self, index):
        filename_line = self.filenames[index] # Scene02/clone/frames/depth/Camera_0/depth_00000.png
        # Get data path
        optical_flow_rel_path = filename_line[0]
        rgb_rel_path = optical_flow_rel_path.replace("flow_2d", "image_clean")

        frame_id = int(rgb_rel_path.split('/')[-1].split('_')[-1].replace('.png', ''))
        next_frame_id = frame_id + 1
        next_rgb_rel_path = rgb_rel_path.replace(f'{frame_id:07d}', f'{next_frame_id:07d}')

        pc_rel_path = rgb_rel_path.replace("image_clean", "pc").replace(".png", ".npz")
        flow_3d_rel_path = rgb_rel_path.replace("image_clean", "flow_3d").replace(".png", ".npy")
        return rgb_rel_path, next_rgb_rel_path, optical_flow_rel_path, pc_rel_path, flow_3d_rel_path
