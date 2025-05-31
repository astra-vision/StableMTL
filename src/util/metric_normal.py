# Author: Bingxin Ke
# Last modified: 2024-02-15


import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F

def mean_angular_error(output, target, valid_mask=None):
    assert output.shape == target.shape, "Output and target must have the same shape."
    assert output.shape[1] == 3, "Output and target must have 3 channels."
    assert len(output.shape) == 4, "Output and target must have 4 dimensions (B, 3, H, W)."
    
    output_normalized = F.normalize(output, p=2, dim=1)
    target_normalized = F.normalize(target, p=2, dim=1)
    err = torch.acos(torch.clamp(torch.sum(target_normalized * output_normalized, dim=1, keepdim=True), -1, 1)) * 180 / np.pi
    
    err *= valid_mask
    n = valid_mask.sum((-1, -2))
    angular_error = err.sum((-1, -2)) / n
    
    
    return angular_error.mean()

