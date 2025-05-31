# Author: Bingxin Ke
# Last modified: 2024-02-19


from pylab import count_nonzero, clip, np
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

# Adapted from https://github.com/apple/ml-hypersim/blob/main/code/python/tools/scene_generate_images_tonemap.py
def tone_map(rgb, entity_id_map):
    assert (entity_id_map != 0).all()

    gamma = 1.0 / 2.2  # standard gamma correction exponent
    inv_gamma = 1.0 / gamma
    percentile = (
        90  # we want this percentile brightness value in the unmodified image...
    )
    brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling

    valid_mask = entity_id_map != -1

    if count_nonzero(valid_mask) == 0:
        scale = 1.0  # if there are no valid pixels, then set scale to 1.0
    else:
        brightness = (
            0.3 * rgb[:, :, 0] + 0.59 * rgb[:, :, 1] + 0.11 * rgb[:, :, 2]
        )  # "CCIR601 YIQ" method for computing brightness
        brightness_valid = brightness[valid_mask]

        eps = 0.0001  # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:
            # Snavely uses the following expression in the code at https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
            # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma - np.log(brightness_nth_percentile_current))
            #
            # Our expression below is equivalent, but is more intuitive, because it follows more directly from the expression:
            # (scale*brightness_nth_percentile_current)^gamma = brightness_nth_percentile_desired

            scale = (
                np.power(brightness_nth_percentile_desired, inv_gamma)
                / brightness_nth_percentile_current
            )

    rgb_color_tm = np.power(np.maximum(scale * rgb, 0), gamma)
    rgb_color_tm = clip(rgb_color_tm, 0, 1)
    return rgb_color_tm


# According to https://github.com/apple/ml-hypersim/issues/9
def dist_2_depth(width, height, flt_focal, distance):
    img_plane_x = (
        np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width)
        .reshape(1, width)
        .repeat(height, 0)
        .astype(np.float32)[:, :, None]
    )
    img_plane_y = (
        np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height)
        .reshape(height, 1)
        .repeat(width, 1)
        .astype(np.float32)[:, :, None]
    )
    img_plane_z = np.full([height, width, 1], flt_focal, np.float32)
    img_plane = np.concatenate([img_plane_x, img_plane_y, img_plane_z], 2)

    depth = distance / np.linalg.norm(img_plane, 2, 2) * flt_focal
    return depth


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


def get_tonemap_scale(rgb_color, valid_mask=None, p=90):
    """Compute the tonemapping scale for an HDR image following the CGIntrinsics
    and Hypersim code-bases. The scale is determined such that the p-th percentile
    value in the input is equal to 0.8 after performing tonemapping.

    params:
        rgb_color (np.array): input rgb values to compute tonemap scale
        p (int) optional: percentile value to map to 0.8 (default 90)

    returns:
        scale (float): scale to multiple by the input before gamma-correction
    """
    gamma = 1.0 / 2.2 # standard gamma correction exponent
    inv_gamma = 1.0 / gamma
    # percentile = 90 # we want this percentile brightness value in the unmodified image...
    brightness_nth_percentile_desired = 0.8 # ...to be this bright after scaling

    brightness       = get_brightness(rgb_color)
    if valid_mask is not None:
        brightness_valid = brightness[valid_mask]
    else:
        brightness_valid = brightness

    # if the kth percentile brightness value in the unmodified image is less than this,
    # set the scale to 0.0 to avoid divide-by-zero
    eps = 0.0001
    brightness_nth_percentile_current = np.percentile(brightness_valid, p)

    if brightness_nth_percentile_current < eps:
        scale = 0.0
    else:
        # Snavely uses the following expression in the code at
        # https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
        # scale = np.exp(
        #           np.log(
        #               brightness_nth_percentile_desired) *
        #               inv_gamma -
        #               np.log(brightness_nth_percentile_current))
        #
        # Our expression below is equivalent, but is more intuitive, because it follows more
        # directly from the expression:
        # (scale*brightness_nth_percentile_current)^gamma = brightness_nth_percentile_desired
        # pylint: disable-next=line-too-long
        scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current

    return scale