import struct

import cv2
import numpy as np


# uMax = 640  # w
# vMax = 480  # h

import torch


def tensor_to_numpy(tensor_in):
    """ torch tensor to numpy array
    """
    if tensor_in is not None:
        if tensor_in.ndim == 3:
            # (C, H, W) -> (H, W, C)
            tensor_in = tensor_in.detach().cpu().permute(1, 2, 0).numpy()
        elif tensor_in.ndim == 4:
            # (B, C, H, W) -> (B, H, W, C)
            tensor_in = tensor_in.detach().cpu().permute(0, 2, 3, 1).numpy()
        else:
            raise Exception('invalid tensor size')
    return tensor_in

def normal_to_rgb(normal, normal_mask=None):
    """ surface normal map to RGB
        (used for visualization)

        NOTE: x, y, z are mapped to R, G, B
        NOTE: [-1, 1] are mapped to [0, 255]
    """
    if torch.is_tensor(normal):
        normal = tensor_to_numpy(normal)
        normal_mask = tensor_to_numpy(normal_mask)

    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal_norm[normal_norm < 1e-12] = 1e-12
    normal = normal / normal_norm

    normal_rgb = (((normal + 1) * 0.5) * 255).astype(np.uint8)
    if normal_mask is not None:
        normal_rgb = normal_rgb * normal_mask     # (B, H, W, 3)
    return normal_rgb


def get_cam_params(calib_path):
    with open(calib_path, 'r') as f:
        data = f.read()
        params = list(map(int, (data.split())))[:-1]
    return params


def get_normal_gt(normal_path):
    # retVal: [-1,1]
    normal_gt = cv2.imread(normal_path, -1)
    normal_gt = normal_gt[:, :, ::-1]
    normal_gt = 1 - normal_gt / 65535 * 2
    return normal_gt


def get_depth(depth_path, height, width):
    with open(depth_path, 'rb') as f:
        data_raw = struct.unpack('f' * width * height, f.read(4 * width * height))
        z = np.array(data_raw).reshape(height, width)

    # create mask, 1 for foreground, 0 for background
    mask = np.ones_like(z)
    mask[z == 1] = 0

    return z, mask


def vector_normalization(normal, eps=1e-8):
    mag = np.linalg.norm(normal, axis=2)
    normal /= (np.expand_dims(mag, axis=2) + eps)
    return normal


def visualization_map_creation(normal, mask):
    mask = np.expand_dims(mask, axis=2)
    vis = normal * mask + mask - 1
    vis = (1 - vis) / 2  # transform the interval from [-1, 1] to [0, 1]
    return vis


def angle_normalization(err_map):
    err_map[err_map > np.pi / 2] = np.pi - err_map[err_map > np.pi / 2]
    return err_map


def evaluation(n_gt, n_est, mask):
    scale = np.pi / 180
    error_map = np.arccos(np.sum(n_gt * n_est, axis=2))
    error_map = angle_normalization(error_map) / scale
    error_map *= mask
    ea = error_map.sum() / mask.sum()
    return error_map, ea

# def softmax(x):
#     x_exp = np.exp(x)
#     x_sum = np.sum(x_exp)
#     return x_exp / x_sum
#
#
# def softmin(x):
#     x_exp = np.exp(-x)
#     x_sum = np.sum(x_exp)
#     return x_exp / x_sum
