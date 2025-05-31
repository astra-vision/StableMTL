import numpy as np
import torch


def get_brightness(rgb, mode='numpy', keep_dim=True):
    """use the CCIR601 YIQ method to compute brightness of an RGB image
    params:
        rgb (np.array or torch.Tensor): RGB image to convert to luminance
        mode (str) optional: whether the input is numpy or torch (default "numpy")
        keep_dim (bool) optional: whether or not to maintain the channel dimension
    returns:
        brightness (np.array or torch.Tensor): single channel image of brightness values
    """
    # "CCIR601 YIQ" method for computing brightness
    if mode == 'numpy':
        brightness = (0.3 * rgb[:,:,0]) + (0.59 * rgb[:,:,1]) + (0.11 * rgb[:,:,2])
        if keep_dim:
            brightness = brightness[:, :, np.newaxis]
    if mode == 'torch':
        brightness = (0.3 * rgb[0,:,:]) + (0.59 * rgb[1,:,:]) + (0.11 * rgb[2, :,:])
        if keep_dim:
            brightness = brightness.unsqueeze(0)
    return brightness



def kitti_benchmark_crop(input_img):
    """
    Crop images to KITTI benchmark size
    Args:
        `input_img` (torch.Tensor): Input image to be cropped.

    Returns:
        torch.Tensor:Cropped image.
    """
    KB_CROP_HEIGHT = 352
    KB_CROP_WIDTH = 1216

    height, width = input_img.shape[-2:]
    top_margin = int(height - KB_CROP_HEIGHT)
    left_margin = int((width - KB_CROP_WIDTH) / 2)
    if 2 == len(input_img.shape):
        out = input_img[
            top_margin : top_margin + KB_CROP_HEIGHT,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            top_margin : top_margin + KB_CROP_HEIGHT,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    return out