# Author: Bingxin Ke
# Last modified: 2024-02-15


import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import numpy as np


class SceneFlowMetrics(object):
    def __init__(self):
        self.reset()
                
    def reset(self):
        self.metrics_3d = {
            'counts': 0.0,
            'EPE3d': 0.0,  # End-point error (3D)
            'acc_strict': 0.0,  # Accuracy with strict threshold
            'acc_relax': 0.0,  # Accuracy with relaxed threshold
            'outliers': 0.0,  # Percentage of outliers
        }
        
    def update(self, scene_flow_preds, scene_flow_gts, valid_masks):
        assert len(scene_flow_gts.shape) == 4, f"Shape scene_flow_gts is {scene_flow_gts.shape} != len(4)"
        assert len(scene_flow_preds.shape) == 4, f"Shape scene_flow_preds is {scene_flow_preds.shape} != len(4)"
        assert len(valid_masks.shape) == 4, "Shape of valid_masks must be (bs, H, W)"
            
        for pred, target, valid_mask in zip(scene_flow_preds, scene_flow_gts, valid_masks):
            # Calculate 3D end-point error
            epe3d_map = torch.sqrt(torch.sum((pred - target) ** 2, dim=0, keepdim=True))
            epe3d_map *= valid_mask
            
            # Calculate flow magnitude
            flow_magnitude = torch.sqrt(torch.sum(target ** 2, dim=0, keepdim=True)) + 1e-4
            
            # Calculate accuracies and outliers
            acc_strict_map = (epe3d_map < 0.05) & (epe3d_map < flow_magnitude * 0.05)
            acc_relax_map = (epe3d_map < 0.1) & (epe3d_map < flow_magnitude * 0.1)
            outlier_map = (epe3d_map > 0.3) & (epe3d_map > flow_magnitude * 0.1)
            
            # Apply valid mask
            acc_strict_map *= valid_mask
            acc_relax_map *= valid_mask
            outlier_map *= valid_mask
            
            # Count valid points
            n = valid_mask.sum().item()
            
            # Update metrics
            self.metrics_3d['EPE3d'] += epe3d_map.sum().item()
            self.metrics_3d['acc_strict'] += acc_strict_map.sum().item()
            self.metrics_3d['acc_relax'] += acc_relax_map.sum().item()
            self.metrics_3d['outliers'] += outlier_map.sum().item()
            self.metrics_3d['counts'] += n
            
    def result(self):
        return {
            'EPE3d': self.metrics_3d['EPE3d'] / self.metrics_3d['counts'],
            'acc_strict': self.metrics_3d['acc_strict'] / self.metrics_3d['counts'],
            'acc_relax': self.metrics_3d['acc_relax'] / self.metrics_3d['counts'],
            'outliers': self.metrics_3d['outliers'] / self.metrics_3d['counts'],
        }
        

class OpticalFlowMetrics(object):
    def __init__(self):
        self.reset()
                
    def reset(self):
        self.metrics_2d = {
            'counts': 0.0, 
            'EPE2d': 0.0, 
            # '1px': 0.0, 
            'Fl': 0.0,
            "angular_error": 0.0,
        }
        
    def update(self, optical_flow_preds, optical_flow_gts, valid_masks):
        
        assert len(optical_flow_gts.shape) == 4, f"Shape optical_flow_gts is {optical_flow_gts.shape} != len(4)"
        assert len(optical_flow_preds.shape) == 4, f"Shape optical_flow_preds is {optical_flow_preds.shape} != len(4)"
        assert len(valid_masks.shape) == 4, "Shape of valid_masks must be (bs, H, W)"
            
        for target, output, valid_mask in zip(optical_flow_gts, optical_flow_preds, valid_masks):
            # import pdb;pdb.set_trace()
            # optical_flow_gt = optical_flow_gt[valid_mask]
            # optical_flow_pred = optical_flow_pred[valid_mask]
            
            
            output_normalized = F.normalize(output, p=2, dim=0)
            target_normalized = F.normalize(target, p=2, dim=0)
            angular_error = torch.acos(torch.clamp(torch.sum(target_normalized * output_normalized, dim=0, keepdim=True), -1, 1)) * 180 / np.pi
            
            angular_error *= valid_mask     
            n = valid_mask.sum()
            angular_error = angular_error.sum()
            
            self.metrics_2d['angular_error'] += angular_error.item()
            self.metrics_2d['counts'] += n.item()
            
            epe2d_map = torch.sqrt(torch.sum((output - target) ** 2, dim=0, keepdim=True))
            epe2d_map *= valid_mask
            epe2d = epe2d_map.sum()
            self.metrics_2d['EPE2d'] += epe2d.item()
            
            flow_2d_target_mag = torch.sqrt(torch.sum(target ** 2, dim=0, keepdim=True))
            fl_err_map = (epe2d_map > 3.0) & (epe2d_map / flow_2d_target_mag > 0.05)
            fl_err_map *= valid_mask
            fl_err = fl_err_map.sum()
            self.metrics_2d['Fl'] += fl_err.item()

            
            # optical_flow_gt = np.transpose(optical_flow_gt, (1, 2, 0)).reshape(-1, 2)
            # valid_mask = np.transpose(valid_mask, (1, 2, 0)).reshape(-1)
            # optical_flow_pred = np.transpose(optical_flow_pred, (1, 2, 0)).reshape(-1, 2)
            
            # optical_flow_gt = optical_flow_gt[valid_mask]
            # optical_flow_pred = optical_flow_pred[valid_mask]
            # epe2d_map = np.sqrt(np.sum((optical_flow_pred - optical_flow_gt) ** 2, axis=1)) 
            
            
            # flow_2d_mask = valid_mask
            # optical_flow_gt_mag = np.linalg.norm(optical_flow_gt, axis=1)
            # # optical_flow_gt_mag[optical_flow_gt_mag == 0] = 1e-3
            # fl_err_map = (epe2d_map > 3.0) & (epe2d_map / optical_flow_gt_mag > 0.05)
            # # fl_err_map = torch.logical_and(epe2d_map > 3.0, epe2d_map / flow_2d_target_mag > 0.05)

            # # self.metrics_2d['counts'] += flow_2d_mask.sum()
            # # self.metrics_2d['EPE2d'] += epe2d_map[flow_2d_mask].sum().item()
            # # self.metrics_2d['EPE2d'] += epe2d_map.sum()
            # # self.metrics_2d['1px'] += np.count_nonzero(epe2d_map[flow_2d_mask] < 1.0)
            # # self.metrics_2d['1px'] += torch.count_nonzero(epe2d_map[valid_mask] < 1.0)
            # # self.metrics_2d['Fl'] += fl_err_map.sum()
            # # self.metrics_2d['Fl'] += f1_err_map[valid_mask].sum()
            

    

    def result(self):
        return {
            'EPE2d': self.metrics_2d['EPE2d'] / self.metrics_2d['counts'],
            # '1px': self.metrics_2d['1px'] / self.metrics_2d['counts'],
            'Fl': self.metrics_2d['Fl'] / self.metrics_2d['counts'],
            "optical_flow_MAE": self.metrics_2d["angular_error"] / self.metrics_2d["counts"],
        }