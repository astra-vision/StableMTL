import os
import tarfile
from io import BytesIO

import numpy as np
import torch
from .semantic import CityscapesEncoder
from PIL import Image
from .base_mtl_dataset import BaseMTLDataset, DepthFileNameMode, DatasetMode
import cv2
from .utils import kitti_benchmark_crop
import torch.nn.functional as F


class KittiFlowDataset(BaseMTLDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        

    def _read_image(self, img_rel_path) -> np.ndarray:
        image_to_read = os.path.join(self.dataset_dir, img_rel_path)
        image = Image.open(image_to_read)  # [H, W, rgb]
        # width, height = image.size
        # image = image.resize((width//2, height//2), Image.NEAREST)
        image = np.asarray(image)
        return image

    def _load_rgb_data(self, rgb_rel_path, next_rgb_rel_path):
        rgb_data = super()._load_rgb_data(rgb_rel_path)
        next_rgb_data = super()._load_rgb_data(next_rgb_rel_path)
        rgb_data['rgb_norm'] = torch.cat([rgb_data['rgb_norm'], next_rgb_data['rgb_norm']], dim=0)
        rgb_data['rgb_int'] = torch.cat([rgb_data['rgb_int'], next_rgb_data['rgb_int']], dim=0)
        rgb_data = {k: kitti_benchmark_crop(v) for k, v in rgb_data.items()}
        return rgb_data


    @staticmethod
    def load_disp_png(filepath):
        array = cv2.imread(filepath, -1)
        valid_mask = array > 0
        disp = array.astype(np.float32) / 256.0
        disp[np.logical_not(valid_mask)] = -1.0
        return disp, valid_mask

    @staticmethod
    def load_calib(filepath):
        with open(filepath) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('P_rect_02'):
                    proj_mat = line.split()[1:]
                    proj_mat = [float(param) for param in proj_mat]
                    proj_mat = np.array(proj_mat, dtype=np.float32).reshape(3, 4)
                    assert proj_mat[0, 1] == proj_mat[1, 0] == 0
                    assert proj_mat[2, 0] == proj_mat[2, 1] == 0
                    assert proj_mat[0, 0] == proj_mat[1, 1]
                    assert proj_mat[2, 2] == 1

        return proj_mat
    
    @staticmethod
    def disp2pc(disp, baseline, f, cx, cy, flow=None):
        h, w = disp.shape
        depth = baseline * f / (disp + 1e-5)

        xx = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
        yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

        if flow is None:
            x = (xx - cx) * depth / f
            y = (yy - cy) * depth / f
        else:
            x = (xx - cx + flow[..., 0]) * depth / f
            y = (yy - cy + flow[..., 1]) * depth / f

        pc = np.concatenate([
            x[:, :, None],
            y[:, :, None],
            depth[:, :, None],
        ], axis=-1)

        return pc
    

    def _get_data_path(self, index):
        optical_flow_rel_path = self.filenames[index][0]
        rgb_rel_path = optical_flow_rel_path.replace("flow_occ/", "image_2/")
        next_rgb_rel_path = rgb_rel_path.replace("_10", "_11")
        
        disp1_rel_path = optical_flow_rel_path.replace("flow_occ/", "disp_occ_0/")
        dip2_rel_path = optical_flow_rel_path.replace("flow_occ/", "disp_occ_1/")
        calibration_rel_path = optical_flow_rel_path.replace("flow_occ/", "calib_cam_to_cam/").replace(".png", ".txt")
        calibration_rel_path = calibration_rel_path.replace("_10", "")
        return rgb_rel_path, next_rgb_rel_path, optical_flow_rel_path, disp1_rel_path, dip2_rel_path, calibration_rel_path
    
    
    def load_optical_flow_png(self, filepath, scale=64.0):
        # for KITTI which uses 16bit PNG images
        # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
        # The -1 is here to specify not to change the image depth (16bit), and is compatible
        # with both OpenCV2 and OpenCV3
        flow_img = cv2.imread(filepath, -1)
        flow = flow_img[:, :, 2:0:-1].astype(np.float32)
        valid_mask = flow_img[:, :, 0] > 0
        flow = flow - 32768.0
        flow = flow / scale
        return flow, valid_mask
    
    def load_scene_flow(self, optical_flow_rel_path, disp1_path, disp2_path, calib_path):
        optical_flow_path = os.path.join(self.dataset_dir, optical_flow_rel_path)
        disp1_path = os.path.join(self.dataset_dir, disp1_path)
        disp2_path = os.path.join(self.dataset_dir, disp2_path)
        calib_path = os.path.join(self.dataset_dir, calib_path)
        
        flow_2d, flow_2d_mask = self.load_optical_flow_png(optical_flow_path)
        disp1, mask1 = self.load_disp_png(disp1_path)
        disp2, mask2 = self.load_disp_png(disp2_path)
        proj_mat = self.load_calib(calib_path)
        f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]
        
        mask = np.logical_and(np.logical_and(mask1, mask2), flow_2d_mask)

        pc1 = self.disp2pc(disp1, 0.54, f, cx, cy)
        pc2 = self.disp2pc(disp2, 0.54, f, cx, cy, flow=flow_2d)
        
        
        scene_flow = pc2 - pc1
        scene_flow = torch.from_numpy(scene_flow).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).bool().unsqueeze(0)
        
        scene_flow = kitti_benchmark_crop(scene_flow)
        mask = kitti_benchmark_crop(mask)
        # scene_flow_norm = F.normalize(scene_flow, p=2, dim=0)
        
        return {
            "scene_flow": scene_flow,
            "valid_mask": mask,
            # "scene_flow_norm": scene_flow_norm
        }
    
    
    def _load_optical_flow_data(self, optical_flow_rel_path):
        optical_flow_path = os.path.join(self.dataset_dir, optical_flow_rel_path)  
        # optical_flow_path = optical_flow_path.replace("flow_occ", "flow_noc")
        optical_flow, optical_flow_valid_mask = self.load_optical_flow_png(optical_flow_path)
        
        optical_flow = torch.from_numpy(optical_flow).float()
        optical_flow_valid_mask = torch.from_numpy(optical_flow_valid_mask).bool()
        
        optical_flow = optical_flow.permute(2, 0, 1)
        optical_flow_valid_mask = optical_flow_valid_mask.unsqueeze(0)
        
        optical_flow = kitti_benchmark_crop(optical_flow)
        optical_flow_valid_mask = kitti_benchmark_crop(optical_flow_valid_mask)
        
        return {
            "valid_mask": optical_flow_valid_mask,
            "optical_flow_raw": optical_flow
        }
    
    def _get_data_item(self, index):
        # Special: depth mask is read from data

        rgb_rel_path, next_rgb_rel_path, optical_flow_rel_path, disp1_rel_path, dip2_rel_path, calibration_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        # rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path, next_rgb_rel_path=next_rgb_rel_path))
        img0_np = self._read_image(rgb_rel_path).transpose(2, 0, 1)
        img1_np = self._read_image(next_rgb_rel_path).transpose(2, 0, 1)
        img0 = torch.from_numpy(img0_np.copy()).float()
        img1 = torch.from_numpy(img1_np.copy()).float()
        
        img0 = kitti_benchmark_crop(img0)
        img1 = kitti_benchmark_crop(img1)
        
        rasters['rgb_int'] = img0
        rasters['rgb_norm'] = img0 / 255.0 * 2.0 - 1.0
        rasters['rgb_next_int'] = img1
        rasters['rgb_next_norm'] = img1 / 255.0 * 2.0 - 1.0
        

        # Depth data
        optical_flow_data = self._load_optical_flow_data(optical_flow_rel_path=optical_flow_rel_path)
        rasters.update(optical_flow_data)
        
        scene_flow_data = self.load_scene_flow(disp1_path=disp1_rel_path, disp2_path=dip2_rel_path, 
                                            calib_path=calibration_rel_path, 
                                            optical_flow_rel_path=optical_flow_rel_path)
        rasters.update(scene_flow_data)

        other = {"index": index, "rgb_relative_path": rgb_rel_path, "optical_flow_rel_path": optical_flow_rel_path}

        return rasters, other

    