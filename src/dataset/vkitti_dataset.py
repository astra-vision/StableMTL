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
from .flow_augmentation import joint_optical_flow_augmentation, joint_scene_flow_augmentation
from .augmentation import joint_depth_augmentation, joint_normal_augmentation, joint_semseg_augmentation, joint_tasks_augmentation
from src.util.alignment import normalize_optical_flow, normalize_scene_flow
import random


class VirtualKITTIDataset(BaseMTLDataset):
    def __init__(
        self,
        kitti_bm_crop,  # Crop to KITTI benchmark size
        valid_mask_crop,  # Evaluation mask. [None, garg or eigen]
        load_semseg_class_ids=False,
        **kwargs,
    ) -> None:
        super().__init__(
            # virtual KITTI data parameter
            min_depth=1e-5,
            max_depth=80,  # 655.35
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )
        self.kitti_bm_crop = kitti_bm_crop
        self.valid_mask_crop = valid_mask_crop
        self.load_semseg_class_ids = load_semseg_class_ids
        assert self.valid_mask_crop in [
            None,
            "garg",  # set evaluation mask according to Garg  ECCV16
            "eigen",  # set evaluation mask according to Eigen NIPS14
        ], f"Unknown crop type: {self.valid_mask_crop}"

        self.semantic_encoder = VKitti2Encoder(n_classes=8)



    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode vKITTI depth
        depth_decoded = depth_in / 100.0
        return depth_decoded

    def _load_rgb_data(self, rgb_rel_path):
        rgb_data = super()._load_rgb_data(rgb_rel_path)
        if self.kitti_bm_crop:
            rgb_data = {
                k: kitti_benchmark_crop(v) for k, v in rgb_data.items()
            }
        return rgb_data


    def _get_valid_mask(self, depth: torch.Tensor):
        # reference: https://github.com/cleinc/bts/blob/master/pytorch/bts_eval.py
        valid_mask = super()._get_valid_mask(depth)  # [1, H, W]

        if self.valid_mask_crop is not None:
            eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
            gt_height, gt_width = eval_mask.shape

            if "garg" == self.valid_mask_crop:
                eval_mask[
                    int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
                ] = 1
            elif "eigen" == self.valid_mask_crop:
                eval_mask[
                    int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                ] = 1

            eval_mask.reshape(valid_mask.shape)
            valid_mask = torch.logical_and(valid_mask, eval_mask)
        return valid_mask


    # ====== stablepoop ======
    def _get_data_item(self, index):
        raise NotImplementedError("Please implement this method in the derived class.")

    def _load_normal(self, normal_rel_path):
        data = np.load(os.path.join(self.dataset_dir, normal_rel_path))  # torch.Size([375, 1242, 3])
        return data


    def _load_optical_flow_data(self, optical_flow_rel_path):
        flow_path = os.path.join(self.dataset_dir, optical_flow_rel_path)
        "Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array"
        # read png to bgr in 16 bit unsigned short

        try:
            bgr = cv2.imread(flow_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if bgr is None:
                return None
            h, w, _c = bgr.shape
        except Exception as e:
            print(f"Error reading optical_flow: {flow_path}")
            raise e

        assert bgr.dtype == np.uint16 and _c == 3
        # b == invalid flow flag == 0 for sky or other invalid flow
        invalid = bgr[..., 0] == 0
        # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 – 1]
        out_flow = 2.0 / (2**16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
        out_flow[..., 0] *= w - 1
        out_flow[..., 1] *= h - 1
        out_flow[invalid] = 0 # or another value (e.g., np.nan)



        # out_flow = out_flow.transpose(2, 0, 1)  # (H, W, 2) -> (2, H, W)
        # out_flow = kitti_benchmark_crop(out_flow)

        valid = ~invalid[:, :, np.newaxis] # (H, W) -> (H, W, 1)
        # valid = kitti_benchmark_crop(valid)
        return {
            # "optical_flow": torch.from_numpy(out_flow).float(),
            # "optical_flow_valid_mask": torch.from_numpy(valid).bool()
            "optical_flow": out_flow,
            "optical_flow_valid_mask": valid
        }


    def _load_semantic_data(self, semantic_rel_path):
        semantic_path = os.path.join(self.dataset_dir, semantic_rel_path)
        semantic_rgb = Image.open(semantic_path)

        semantic_original_rgb = np.asarray(semantic_rgb, dtype=np.uint8)

        # class_id: 0 -> 7
        semantic_class_id = self.semantic_encoder.encode_segmap(semantic_original_rgb)

        if self.load_semseg_class_ids:
            semantic_map = semantic_class_id[:,:,None] # unsqueeze dim=2 for channel dim
        else:
            semantic_rgb_int = self.semantic_encoder.segmap2color(semantic_class_id)
            semantic_map = semantic_rgb_int / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

        semantic_valid_mask = semantic_class_id != self.semantic_encoder.ignore_index
        semantic_valid_mask = semantic_valid_mask[..., np.newaxis]
        return semantic_map, semantic_valid_mask


    def _convert_to_vae_space(self, data):
        raise NotImplementedError("Please implement this method in the derived class.")

    def _load_normal_data(self, normal_rel_path):
        outputs = {}
        normal_data = np.load(os.path.join(self.dataset_dir, normal_rel_path))  # torch.Size([375, 1242, 3])
        # torch.Size([3, 375, 1242])
        normal_data = normal_data.transpose(2, 0, 1)
        normal_data = kitti_benchmark_crop(normal_data) # (3, 352, 1216)
        normal_data = torch.from_numpy(normal_data).float()  # torch.Size([3, 352, 1216])
        outputs["normal"] = normal_data
        return outputs

    def _load_scene_flow(self, scene_flow_rel_path):
        flow_path = os.path.join(self.dataset_dir, scene_flow_rel_path)
        "Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array"
        # read png to bgr in 16 bit unsigned short

        try:
            bgr = cv2.imread(flow_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if bgr is None:
                return None
            h, w, _c = bgr.shape
        except Exception as e:
            print(f"Error reading optical_flow: {flow_path}")
            raise e

        assert bgr.dtype == np.uint16 and _c == 3
        out_flow = 2.0 / (2**16 - 1.0) * bgr[..., ::-1].astype('f4') - 1
        out_flow *= 10.0

        return out_flow

    def _get_data_path(self, index):
        raise NotImplementedError("Please implement this method in the derived class.")




class VirtualKITTIDepthDataset(VirtualKITTIDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_type = "depth"

    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path = self._get_data_path(index=index)


        # rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))
        img0 = self._read_image(rgb_rel_path) # (H, W, 3)
        depth_raw = self._read_depth_file(depth_rel_path)[..., np.newaxis] # (H, W, 1)
        valid_mask = self._get_valid_mask(depth_raw) # (H, W, 1)

        img0, depth_raw, valid_mask = joint_depth_augmentation(
            img0, depth_raw, valid_mask, self.augm_args['vkitti_depth']
        )

        img0 = img0.transpose(2, 0, 1)
        depth_raw = depth_raw.transpose(2, 0, 1)
        valid_mask = valid_mask.transpose(2, 0, 1)

        img0 = torch.from_numpy(img0.copy()).float()
        depth_raw = torch.from_numpy(depth_raw.copy()).float()
        valid_mask = torch.from_numpy(valid_mask.copy()).bool()

        if DatasetMode.EVAL == self.mode:
            img0 = kitti_benchmark_crop(img0)
            depth_raw = kitti_benchmark_crop(depth_raw)
            valid_mask = kitti_benchmark_crop(valid_mask)

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
        filename_line = self.filenames[index] # Scene02/clone/frames/depth/Camera_0/depth_00000.png

        # Get data path
        depth_rel_path = filename_line[0] # original/Scene02/clone/frames/rgb/Camera_0/rgb_00000.png
        rgb_rel_path = depth_rel_path.replace('depth', 'rgb').replace('png', 'jpg')

        return rgb_rel_path, depth_rel_path



class VirtualKITTINormalDataset(VirtualKITTIDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_type = "normal"

    def _load_normal(self, normal_rel_path):
        data = np.load(os.path.join(self.dataset_dir, normal_rel_path))  # torch.Size([375, 1242, 3])
        return data

    def _get_data_item(self, index):
        rgb_rel_path, normal_rel_path, depth_rel_path = self._get_data_path(index=index)

        img0 = self._read_image(rgb_rel_path) # (H, W, 3)
        normal = self._load_normal(normal_rel_path=normal_rel_path) # (H, W, 3)
        depth = self._read_depth_file(depth_rel_path)[..., np.newaxis] # (H, W, 1)
        valid_mask = self._get_valid_mask(depth) # (H, W, 1)

        img0, normal, valid_mask = joint_normal_augmentation(
            img0, normal, valid_mask, self.augm_args['vkitti_normal']
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
        filename_line = self.filenames[index] # Scene02/clone/frames/depth/Camera_0/depth_00000.png

        # Get data path
        normal_rel_path = filename_line[0]
        rgb_rel_path = normal_rel_path.replace('normal_estimated', 'original').replace('normal', 'rgb').replace('npy', 'jpg')
        depth_rel_path = rgb_rel_path.replace('rgb', 'depth').replace('jpg', 'png')
        return rgb_rel_path, normal_rel_path, depth_rel_path


class VirtualKITTISemsegDataset(VirtualKITTIDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_type = "semantic"



    def _get_data_item(self, index):
        rgb_rel_path, semantic_rel_path = self._get_data_path(index=index)

        rasters = {}
        img0 = self._read_image(rgb_rel_path)
        semantic_rgb_norm, semantic_valid_mask = self._load_semantic_data(
            semantic_rel_path=semantic_rel_path
        )
        img0, semantic_rgb_norm, semantic_valid_mask = joint_semseg_augmentation(
            img0, semantic_rgb_norm, semantic_valid_mask, self.augm_args['vkitti_semseg']
        )

        img0 = img0.transpose(2, 0, 1)
        semantic_rgb_norm = semantic_rgb_norm.transpose(2, 0, 1)
        semantic_valid_mask = semantic_valid_mask.transpose(2, 0, 1)

        img0 = torch.from_numpy(img0.copy()).float()
        semantic_rgb_norm = torch.from_numpy(semantic_rgb_norm.copy()).float()
        semantic_valid_mask = torch.from_numpy(semantic_valid_mask.copy()).bool()

        rasters['rgb_int'] = img0
        rasters['rgb_norm'] = img0 / 255.0 * 2.0 - 1.0

        rasters[DatasetConst.OUTPUT_FIELD] = semantic_rgb_norm
        rasters[DatasetConst.VALID_MASK_FIELD] = semantic_valid_mask
        other = {"index": index, "rgb_relative_path": rgb_rel_path,
                 DatasetConst.OUTPUT_TYPE_FIELD: "semantic"}
        return rasters, other


    def _get_data_path(self, index):
        filename_line = self.filenames[index] # Scene02/clone/frames/depth/Camera_0/depth_00000.png

        # Get data path
        semantic_rel_path = filename_line[0]
        rgb_rel_path = semantic_rel_path.replace('classSegmentation', 'rgb').replace('png', 'jpg').replace('classgt', 'rgb')

        return rgb_rel_path, semantic_rel_path


class VirtualKITTIOpticalFlowDataset(VirtualKITTIDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_type = "optical_flow"
        # self.optical_flow_transform =  ScaleShiftOpticalFlowNormalizer()

    def _load_rgb_data(self, rgb_rel_path, next_rgb_rel_path):
        rgb_data = super()._load_rgb_data(rgb_rel_path)
        next_rgb_data = super()._load_rgb_data(next_rgb_rel_path)

        rgb_data['rgb_norm'] = torch.cat([rgb_data['rgb_norm'], next_rgb_data['rgb_norm']], dim=0)
        rgb_data['rgb_int'] = torch.cat([rgb_data['rgb_int'], next_rgb_data['rgb_int']], dim=0)
        return rgb_data

    def _get_data_item(self, index):
        rgb_rel_path, next_rgb_rel_path, optical_flow_rel_path = self._get_data_path(index=index)

        rasters = {}
        optical_flow_data = self._load_optical_flow_data(
            optical_flow_rel_path=optical_flow_rel_path
        )
        optical_flow = optical_flow_data["optical_flow"]
        optical_flow_valid_mask = optical_flow_data["optical_flow_valid_mask"]

        img0 = self._read_image(rgb_rel_path)
        img1 = self._read_image(next_rgb_rel_path)

        if DatasetMode.TRAIN == self.mode:
            img0, img1, optical_flow, optical_flow_valid_mask = joint_optical_flow_augmentation(
                img0, img1, optical_flow, optical_flow_valid_mask, self.augm_args['vkitti_flow']
            )


        img0 = img0.transpose(2, 0, 1)
        img1 = img1.transpose(2, 0, 1)
        optical_flow = optical_flow.transpose(2, 0, 1)
        optical_flow_valid_mask = optical_flow_valid_mask.transpose(2, 0, 1)

        img0 = torch.from_numpy(img0).float()
        img1 = torch.from_numpy(img1).float()
        optical_flow = torch.from_numpy(optical_flow).float()
        optical_flow_valid_mask = torch.from_numpy(optical_flow_valid_mask).bool()


        if DatasetMode.EVAL == self.mode:
            img0 = kitti_benchmark_crop(img0)
            img1 = kitti_benchmark_crop(img1)
            optical_flow = kitti_benchmark_crop(optical_flow)
            optical_flow_valid_mask = kitti_benchmark_crop(optical_flow_valid_mask)


        rasters['rgb_int'] = img0
        rasters['rgb_norm'] = img0 / 255.0 * 2.0 - 1.0

        rasters['rgb_next_int'] = img1
        rasters['rgb_next_norm'] = img1 / 255.0 * 2.0 - 1.0

        norm_type = "hw"
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
        rgb_rel_path = optical_flow_rel_path.replace('forwardFlow', 'rgb').replace('png', 'jpg').replace('flow', 'rgb')
        frame_id = int(rgb_rel_path.split('/')[-1].split('_')[-1].replace('.jpg', ''))
        next_frame_id = frame_id + 1
        next_rgb_rel_path = rgb_rel_path.replace(f'_{frame_id:05d}', f'_{next_frame_id:05d}')
        return rgb_rel_path, next_rgb_rel_path, optical_flow_rel_path



class VirtualKITTISceneFlowDataset(VirtualKITTIDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


    def _load_scene_flow(self, scene_flow_rel_path):
        flow_path = os.path.join(self.dataset_dir, scene_flow_rel_path)
        "Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array"
        # read png to bgr in 16 bit unsigned short

        try:
            bgr = cv2.imread(flow_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if bgr is None:
                return None
            h, w, _c = bgr.shape
        except Exception as e:
            print(f"Error reading optical_flow: {flow_path}")
            raise e

        assert bgr.dtype == np.uint16 and _c == 3
        out_flow = 2.0 / (2**16 - 1.0) * bgr[..., ::-1].astype('f4') - 1
        out_flow *= 10.0

        return out_flow



    def _get_data_item(self, index):
        rgb_rel_path, next_rgb_rel_path, optical_flow_rel_path, scene_flow_rel_path= self._get_data_path(index=index)


        rasters = {}
        img0 = self._read_image(rgb_rel_path)
        img1 = self._read_image(next_rgb_rel_path)
        scene_flow = self._load_scene_flow(
            scene_flow_rel_path=scene_flow_rel_path
        )

        optical_flow_data = self._load_optical_flow_data(
            optical_flow_rel_path=optical_flow_rel_path
        )
        flow_valid_mask = optical_flow_data["optical_flow_valid_mask"]

        if DatasetMode.TRAIN == self.mode:
            img0, img1, scene_flow, flow_valid_mask = joint_scene_flow_augmentation(
                img0, img1, scene_flow, flow_valid_mask, self.augm_args['vkitti_scene_flow']
            )

        img0 = img0.transpose(2, 0, 1)
        img1 = img1.transpose(2, 0, 1)
        scene_flow = scene_flow.transpose(2, 0, 1)
        flow_valid_mask = flow_valid_mask.transpose(2, 0, 1)

        img0 = torch.from_numpy(img0.copy()).float()
        img1 = torch.from_numpy(img1.copy()).float()
        scene_flow = torch.from_numpy(scene_flow.copy()).float()
        flow_valid_mask = torch.from_numpy(flow_valid_mask.copy()).bool()

        norm_type = "hw"
        scene_flow_norm = normalize_scene_flow(scene_flow, type=norm_type)


        rasters['rgb_int'] = img0
        rasters['rgb_norm'] = img0 / 255.0 * 2.0 - 1.0

        rasters['rgb_next_int'] = img1
        rasters['rgb_next_norm'] = img1 / 255.0 * 2.0 - 1.0


        rasters[DatasetConst.OUTPUT_FIELD] = scene_flow_norm
        rasters[DatasetConst.VALID_MASK_FIELD] = flow_valid_mask
        rasters['scene_flow'] = scene_flow

        other = {"index": index, "rgb_relative_path": rgb_rel_path,
                 DatasetConst.OUTPUT_TYPE_FIELD: "scene_flow"}

        return rasters, other


    def _get_data_path(self, index):
        filename_line = self.filenames[index] # Scene02/clone/frames/depth/Camera_0/depth_00000.png

        scene_flow_rel_path = filename_line[0]

        rgb_rel_path = scene_flow_rel_path.replace('forwardSceneFlow', 'rgb').replace('png', 'jpg').replace('sceneFlow', 'rgb')
        optical_flow_rel_path = scene_flow_rel_path.replace('forwardSceneFlow', 'forwardFlow').replace('sceneFlow', 'flow')

        frame_id = int(rgb_rel_path.split('/')[-1].split('_')[-1].replace('.jpg', ''))
        next_frame_id = frame_id + 1
        next_rgb_rel_path = rgb_rel_path.replace(f'_{frame_id:05d}', f'_{next_frame_id:05d}')
        return rgb_rel_path, next_rgb_rel_path, optical_flow_rel_path, scene_flow_rel_path

