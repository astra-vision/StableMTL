from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, Normalize
from matplotlib import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torchvision
import torch


def tone_map(brightness):
    # assert (entity_id_map != 0).all()

    gamma = 1.0 / 2.2  # standard gamma correction exponent
    inv_gamma = 1.0 / gamma
    percentile = (
        90  # we want this percentile brightness value in the unmodified image...
    )
    brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling

    valid_mask = np.ones_like(brightness)

    if np.count_nonzero(valid_mask) == 0:
        scale = 1.0  # if there are no valid pixels, then set scale to 1.0
    else:
        # brightness = (
        #     0.3 * rgb[:, :, 0] + 0.59 * rgb[:, :, 1] + 0.11 * rgb[:, :, 2]
        # )  # "CCIR601 YIQ" method for computing brightness
        # brightness_valid = brightness[valid_mask]
        brightness_valid = brightness

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
    brightness = np.power(np.maximum(scale * brightness, 0), gamma)
    brightness = np.clip(brightness, 0, 1)
    return brightness


def map_class_to_color(segmentation, class_color_embeddings):
    """Map class IDs in the segmentation to RGB colors using the class_color_embeddings array."""
    # Initialize an empty array for the colored segmentation
    height, width = segmentation.shape
    colored_segmentation = np.zeros((height, width, 3), dtype=np.uint8)

    # Map each class ID to its corresponding color
    for class_id in range(class_color_embeddings.shape[0]):
        mask = segmentation == class_id
        colored_segmentation[mask] = class_color_embeddings[class_id]

    return colored_segmentation
    
    
def visualize_semantic(semantic_class_id_pred, semantic_class_id_gt, class_color_embeddings, png_save_path):
    
    # Generate color images for the predicted and ground truth segmentations
    pred_colored = map_class_to_color(semantic_class_id_pred, class_color_embeddings)
    gt_colored = map_class_to_color(semantic_class_id_gt, class_color_embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Display predicted segmentation
    axes[0].imshow(pred_colored)
    axes[0].set_title('Predicted Segmentation')
    axes[0].axis('off')

    # Display ground truth segmentation
    axes[1].imshow(gt_colored)
    axes[1].set_title('Ground Truth Segmentation')
    axes[1].axis('off')

    # Save the figure
    plt.tight_layout()
    plt.savefig(png_save_path)
    
def visualize_semantic_pred_only(semantic_class_id_pred, class_color_embeddings, png_save_path):
    """Visualizes semantic segmentation prediction only by mapping class IDs to colors"""
    # Generate color image for the predicted segmentation
    pred_colored = map_class_to_color(semantic_class_id_pred, class_color_embeddings)

    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Display predicted segmentation
    ax.imshow(pred_colored)
    # ax.set_title('Predicted Segmentation')
    ax.axis('off')

    # Save the figure
    plt.tight_layout() 
    plt.savefig(png_save_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved to {png_save_path}")
    plt.close(fig)
    
    

    
def visualize_scene_flow(rgb_img, flow_pred, flow_gt, valid_mask, png_save_path, next_rgb_img=None):
    rgb_img = np.transpose(rgb_img, (1, 2, 0))
    flow_pred = np.transpose(flow_pred, (1, 2, 0))  # H, W, 3
    flow_gt = np.transpose(flow_gt, (1, 2, 0))      # H, W, 3
    valid_mask = np.transpose(valid_mask, (1, 2, 0)).squeeze()
    
    flow_pred_norm = np.linalg.norm(flow_pred, axis=2, keepdims=True) 
    flow_gt_norm = np.linalg.norm(flow_gt, axis=2, keepdims=True)
    max_norm = max(flow_pred_norm.max(), flow_gt_norm.max())
    
    flow_pred_color = (flow_pred / max_norm + 1) / 2.0 
    flow_gt_color = (flow_gt / max_norm + 1) / 2.0
    
    flow_gt_color = flow_gt_color * valid_mask[:, :, np.newaxis]
    
    epe = np.sqrt(np.sum((flow_pred - flow_gt) ** 2, axis=2))

    cmap = plt.cm.Reds.copy()
    min_epe = np.min(epe[valid_mask])
    max_epe = np.max(epe[valid_mask])
    norm = Normalize(vmin=min_epe, vmax=max_epe)
    epe_norm = norm(epe)
    epe_masked_color = cmap(epe_norm)
    epe_masked_color[~valid_mask] = [0, 0, 0, 1]
    # cmap.set_bad(color='black')  # Set masked values (NaN) to black
    
    # Create a 2x2 figure layout
    fig, axes = plt.subplots(3, 4, figsize=(18, 8))
    
    # Display RGB image
    axes[0, 0].imshow(rgb_img[:,:, :3])
    axes[0, 0].set_title('Input')
    axes[0, 0].axis('off')
    
    if next_rgb_img is not None:
        axes[0, 1].imshow(next_rgb_img[:,:, :3])    
        axes[0, 1].set_title('Input')
        axes[0, 1].axis('off')
    
    # Display predicted flow
    axes[0, 2].imshow(flow_pred_color)
    axes[0, 2].set_title('Predicted Flow')
    axes[0, 2].axis('off')
    
    # Display ground truth flow
    axes[0, 3].imshow(flow_gt_color)
    axes[0, 3].set_title('Ground Truth Flow')
    axes[0, 3].axis('off')
    
    # Plot histogram of valid epe values
    axes[1, 0].hist(epe[valid_mask], bins=50,  alpha=0.7)
    axes[1, 0].set_title('EPE Distribution')
    axes[1, 0].set_xlabel('EPE Value')
    axes[1, 0].set_ylabel('Frequency')
    
    
    # Display epe map with masked pixels in black
    im = axes[2, 0].imshow(epe_masked_color)
    axes[2, 0].set_title('EPE')
    axes[2, 0].axis('off')
    
    # Adjust the subplot parameters to make room for the colorbar
    plt.subplots_adjust(bottom=0.3)

    # Create a new axes for the colorbar at the bottom
    cbar_ax = fig.add_axes([0.25, 0.00, 0.5, 0.02])

    # Create the colorbar in the new axes
    cbar = colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
    cbar.set_label('Error Magnitude')

    # Plot histogram of valid ae values
    axes[1, 1].hist(flow_gt[:,:, 0][valid_mask], bins=50,  alpha=0.7)
    axes[1, 1].set_title('Gt flow x Distribution')
    axes[1, 1].set_xlabel('flow Value')
    axes[1, 1].set_ylabel('Frequency')
    
    axes[1, 2].hist(flow_gt[:,:, 1][valid_mask], bins=50,  alpha=0.7)
    axes[1, 2].set_title('Gt flow y Distribution')
    axes[1, 2].set_xlabel('flow Value')
    axes[1, 2].set_ylabel('Frequency')
    
    # Plot histogram of valid ae values
    axes[2, 1].hist(flow_pred[:,:, 0][valid_mask], bins=50,  alpha=0.7)
    axes[2, 1].set_title('Predicted flow x Distribution')
    axes[2, 1].set_xlabel('flow Value')
    axes[2, 1].set_ylabel('Frequency')
    
    axes[2, 2].hist(flow_pred[:,:, 1][valid_mask], bins=50,  alpha=0.7)
    axes[2, 2].set_title('Predicted flow y Distribution')
    axes[2, 2].set_xlabel('flow Value')
    axes[2, 2].set_ylabel('Frequency')
    
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(png_save_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved to {png_save_path}")
    plt.close(fig)


def visualize_scene_flow_pred_only(flow_pred, png_save_path):
    """Visualizes scene flow prediction with split visualization: XY as colorwheel, Z as colormap"""
    flow_pred = np.transpose(flow_pred, (1, 2, 0))  # H, W, 3
    # XY colorwheel encoding (like optical flow)
    xy = flow_pred[..., :2]
    mag = np.linalg.norm(xy, axis=2)
    # ang = np.arctan2(xy[..., 1], xy[..., 0])
    ang = np.arctan2(-xy[..., 1], -xy[..., 0])
    hsv = np.zeros((*xy.shape[:2], 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)  # Hue: angle normalized to [0,1]
    # Scale magnitude to custom range
    mag_min, mag_max = mag.min(), mag.max()
    # mag_min, mag_max = np.percentile(mag, 2), np.percentile(mag, 98)
    mag_target_min, mag_target_max = 0.0, 1.0  # Custom target range for magnitude
    mag = mag_target_min + (mag - mag_min) * (mag_target_max - mag_target_min) / (mag_max - mag_min + 1e-6)
    hsv[..., 1] = np.clip(mag, 0.0, 1.0)
    
    # Scale Z component to custom range
    z = flow_pred[..., 2] * -1
    z_min, z_max = z.min(), z.max()
    # # z_min, z_max = np.percentile(z, 2), np.percentile(z, 98)
    z_target_min, z_target_max = 0.0, 1.0  # Custom target range for Z
    z = z_target_min + (z - z_min) * (z_target_max - z_target_min) / (z_max - z_min + 1e-6)
    # z = z * 0.5 + 0.5
    hsv[..., 2] = np.clip(z, 0.0, 1.0)
    xy_color = hsv_to_rgb(hsv)

    # Z encoding: diverging colormap (e.g., RdBu)
    # z = flow_pred[..., 2]
    # z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)  # Normalize to [0,1]
    # z_color = plt.get_cmap('RdBu_r')(z_norm)[..., :3]  # Use only RGB, drop alpha

    # Plot single image
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(xy_color)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(png_save_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved to {png_save_path}")

    
def visualize_optical_flow_pred_only(flow_pred, png_save_path, max_flow=512):
    flow_pred = np.transpose(flow_pred, (1, 2, 0))  # H, W, 2
    
    
    # Generate colored flow images
    flow_pred_ts = torch.from_numpy(flow_pred).permute(2, 0, 1)    
    flow_pred_color = torchvision.utils.flow_to_image(flow_pred_ts)
    flow_pred_color = flow_pred_color.permute(1, 2, 0).numpy()
    
    # Create a figure and plot flow visualization
    fig, ax = plt.subplots(1, 1)
    ax.imshow(flow_pred_color)
    # ax.set_title('Predicted Flow')
    ax.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(png_save_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved image to {png_save_path}")
    # print(f"Saved flow to {npy_save_path}")
    plt.close(fig)

def visualize_optical_flow(rgb_img, flow_pred, flow_gt, valid_mask, png_save_path, max_flow=512, next_rgb_img=None):
    rgb_img = np.transpose(rgb_img, (1, 2, 0))
    if next_rgb_img is not None:
        next_rgb_img = np.transpose(next_rgb_img, (1, 2, 0))
    flow_pred = np.transpose(flow_pred, (1, 2, 0))  # H, W, 2
    flow_gt = np.transpose(flow_gt, (1, 2, 0))      # H, W, 2
    valid_mask = np.transpose(valid_mask, (1, 2, 0)).squeeze()
    
    
    # Generate colored flow images
    # flow_pred_color = flow_to_image(flow_pred)
    # flow_gt_color = flow_to_image(flow_gt)
    flow_pred_ts = torch.from_numpy(flow_pred).permute(2, 0, 1)
    flow_gt_ts = torch.from_numpy(flow_gt).permute(2, 0, 1)
    
    flow_pred_color = torchvision.utils.flow_to_image(flow_pred_ts)
    flow_gt_color = torchvision.utils.flow_to_image(flow_gt_ts)
    flow_pred_color = flow_pred_color.permute(1, 2, 0).numpy()
    flow_gt_color = flow_gt_color.permute(1, 2, 0).numpy()
    flow_gt_color = flow_gt_color * valid_mask[:, :, np.newaxis]
    
    # Compute End-Point Error (EPE)
    epe = np.sqrt(np.sum((flow_pred - flow_gt) ** 2, axis=2))
    ae = np.arccos(np.clip(np.sum(flow_pred * flow_gt, axis=2), -1, 1))  * 180 / np.pi
    # Create a masked array for EPE
    # epe_masked = np.ma.array(epe, mask=valid_mask == 0)
    # epe_masked = epe.copy()
    # epe_masked[~valid_mask] = 0
    # epe_masked = np.ma.array(epe, mask=~valid_mask)
    cmap = plt.cm.Reds.copy()
    min_epe = np.min(epe[valid_mask])
    max_epe = np.max(epe[valid_mask])
    norm = Normalize(vmin=min_epe, vmax=max_epe)
    epe_norm = norm(epe)
    epe_masked_color = cmap(epe_norm)
    epe_masked_color[~valid_mask] = [0, 0, 0, 1]

    # import pdb;pdb.set_trace()
    # epe_masked[valid_mask] = 1
    # import pdb;pdb.set_trace()
    # Create a custom colormap with black for masked values
    # cmap = plt.cm.Reds.copy()
    # min_ae = np.min(ae[valid_mask])
    # max_ae = np.max(ae[valid_mask])
    # norm = Normalize(vmin=min_ae, vmax=max_ae)
    # ae_norm = norm(ae)
    # ae_masked_color = cmap(ae_norm)
    # ae_masked_color[~valid_mask] = [0, 0, 0, 1]
    # cmap.set_bad(color='black')  # Set masked values (NaN) to black
    
    # Create a 3x3 figure layout
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    
    # Display #1 RGB image
    axes[0, 0].imshow(rgb_img[:,:, :3].astype(np.uint8))
    axes[0, 0].set_title('Input')
    axes[0, 0].axis('off')
    
    # Display #2 RGB image
    if next_rgb_img is not None:
        axes[0, 1].imshow(next_rgb_img[:,:, :3].astype(np.uint8))
        axes[0, 1].set_title('Input')
        axes[0, 1].axis('off')

    
    # Display predicted flow
    axes[0, 2].imshow(flow_pred_color)
    axes[0, 2].set_title('Predicted Flow')
    axes[0, 2].axis('off')
    
    # Display #4 ground truth flow
    axes[0, 3].imshow(flow_gt_color)
    axes[0, 3].set_title('Ground Truth Flow')
    axes[0, 3].axis('off')
    
    # Plot histogram of valid ae values
    # axes[1, 0].hist(ae[valid_mask], bins=50,  alpha=0.7)
    # axes[1, 0].set_title('AE Distribution')
    # axes[1, 0].set_xlabel('AE Value')
    # axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 3].imshow(epe_masked_color)
    axes[1, 3].set_title('EPE')
    axes[1, 3].axis('off')
    
    axes[2, 3].hist(epe[valid_mask], bins=50,  alpha=0.7)
    axes[2, 3].set_title('EPE Distribution')
    axes[2, 3].set_xlabel('EPE Value')
    axes[2, 3].set_ylabel('Frequency')
    
    
    # Display ae map with masked pixels in black
    # im = axes[2, 0].imshow(ae_masked_color)
    # axes[2, 0].set_title('Angular Error (ae)')
    # axes[2, 0].axis('off')
    
    # Adjust the subplot parameters to make room for the colorbar
    plt.subplots_adjust(bottom=0.3)

    # Create a new axes for the colorbar at the bottom
    cbar_ax = fig.add_axes([0.25, 0.00, 0.5, 0.02])

    # Create the colorbar in the new axes
    cbar = colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
    cbar.set_label('Error Magnitude')

    # Plot histogram of valid ae values
    axes[1, 1].hist(flow_gt[:,:, 0][valid_mask], bins=50,  alpha=0.7)
    axes[1, 1].set_title('Gt flow x Distribution')
    axes[1, 1].set_xlabel('flow Value')
    axes[1, 1].set_ylabel('Frequency')
    
    axes[1, 2].hist(flow_gt[:,:, 1][valid_mask], bins=50,  alpha=0.7)
    axes[1, 2].set_title('Gt flow y Distribution')
    axes[1, 2].set_xlabel('flow Value')
    axes[1, 2].set_ylabel('Frequency')
    
    # Plot histogram of valid ae values
    axes[2, 1].hist(flow_pred[:,:, 0][valid_mask], bins=50,  alpha=0.7)
    axes[2, 1].set_title('Predicted flow x Distribution')
    axes[2, 1].set_xlabel('flow Value')
    axes[2, 1].set_ylabel('Frequency')
    
    axes[2, 2].hist(flow_pred[:,:, 1][valid_mask], bins=50,  alpha=0.7)
    axes[2, 2].set_title('Predicted flow y Distribution')
    axes[2, 2].set_xlabel('flow Value')
    axes[2, 2].set_ylabel('Frequency')
    
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(png_save_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved to {png_save_path}")
    plt.close(fig)

    
    
# def visualize_optical_flow(rgb_img, flow_pred, flow_gt, valid_mask, png_save_path, max_flow=512):
#     rgb_img = np.transpose(rgb_img, (1, 2, 0))
#     flow_pred = np.transpose(flow_pred, (1, 2, 0)) # H, W, 2
#     flow_gt = np.transpose(flow_gt, (1, 2, 0)) # H, W, 2
#     valid_mask = np.transpose(valid_mask, (1, 2, 0))
#     # max_flow = np.max([np.max(flow_pred), np.max(flow_gt),np.abs(np.min(flow_pred)), np.abs(np.min(flow_gt))])
#     # max_flow = np.max([np.max(flow_gt), np.abs(np.min(flow_gt))])
#     # flow_pred_color = viz_optical_flow(flow_pred, max_flow)
#     # flow_gt_color = viz_optical_flow(flow_gt, max_flow)

    
#     flow_pred_color = flow_to_image(flow_pred)
#     # flow_pred_color = flow_pred_color * valid_mask
#     flow_gt_color = flow_to_image(flow_gt)
#     flow_gt_color = flow_gt_color * valid_mask
    
#     fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
#     # Display predicted segmentation
#     axes[0].imshow(rgb_img)
#     axes[0].set_title('Input')
#     axes[0].axis('off')

#     # Display ground truth segmentation
#     axes[1].imshow(flow_pred_color)
#     axes[1].set_title('Pred')
#     axes[1].axis('off')
    
#     # Display ground truth segmentatioqn
#     axes[2].imshow(flow_gt_color)
#     axes[2].set_title('GT')
#     axes[2].axis('off')

    
#     # Save the figure
#     plt.tight_layout()
#     plt.savefig(png_save_path, bbox_inches='tight', pad_inches=0)
    
def viz_optical_flow(flow, max_flow=512):
    n = 8
    u, v = flow[0, :, :], flow[0, :, :]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)

    image_h = np.mod(angle / (2 * np.pi) + 1, 1)
    image_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    image_v = np.ones_like(image_s)

    image_hsv = np.stack([image_h, image_s, image_v], axis=2)
    image_rgb = hsv_to_rgb(image_hsv)
    image_rgb = np.uint8(image_rgb * 255)

    return image_rgb


# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03


def make_colorwheel():
    """
    Generates a color wheel for optical_flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for optical_flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def colors_to_flow_uv(flow_image, convert_to_bgr=False):
    """
    Convert a flow visualization image back to UV flow components.
    
    Args:
        flow_image (np.ndarray): Flow visualization image of shape [H,W,3]
        convert_to_bgr (bool, optional): Whether input image is in BGR format. Defaults to False.

    Returns:
        np.ndarray: Flow UV image of shape [H,W,2]
    """
    assert flow_image.ndim == 3, 'input image must have three dimensions'
    assert flow_image.shape[2] == 3, 'input image must have shape [H,W,3]'
    
    if convert_to_bgr:
        flow_image = flow_image[...,::-1]
        
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    
    # Normalize image values to [0,1]
    flow_image = flow_image.astype(np.float32) / 255.0
    
    # Get hue and saturation from image
    hsv = np.zeros_like(flow_image)
    hsv[...,1] = np.sqrt(np.sum(flow_image[...,:2]**2, axis=2))
    hsv[...,0] = np.arctan2(flow_image[...,1], flow_image[...,0]) * 180 / np.pi / 2
    hsv[...,0] = (hsv[...,0] + 180) % 180
    
    # Convert hue to flow direction
    a = (hsv[...,0] / 180.0 * 2 - 1) * np.pi
    
    # Get magnitude from saturation
    rad = 1 - hsv[...,1]
    rad[rad < 0.75] = rad[rad < 0.75] / 0.75
    rad[rad >= 0.75] = 0
    
    # Convert to u,v components
    u = -rad * np.cos(a) 
    v = -rad * np.sin(a)
    
    flow_uv = np.stack([u, v], axis=2)
    
    return flow_uv

def visualize_depth(depth, png_save_path, valid_mask=None, cmap='Spectral'):
    """Visualize a depth map with optional masking and save as PNG.
    Args:
        depth: numpy array (H, W), (1, H, W), or (H, W, 1)
        png_save_path: output file path
        valid_mask: optional numpy array (H, W), bool or 0/1, where True/1 = valid
        cmap: matplotlib colormap name (default: 'plasma')
    """
    # Squeeze to (H, W)
    if depth.ndim == 3:
        depth = np.squeeze(depth)
    # Mask invalid values if provided
    if valid_mask is not None:
        depth_vis = np.copy(depth)
        depth_vis[~valid_mask] = np.nan
    else:
        depth_vis = depth
    # Normalize ignoring NaNs
    vmin = np.nanmin(depth_vis)
    vmax = np.nanmax(depth_vis)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    depth_norm = norm(depth_vis)
    # Apply colormap
    depth_color = plt.get_cmap(cmap)(depth_norm)
    # Remove alpha channel if present
    if depth_color.shape[-1] == 4:
        depth_color = depth_color[..., :3]
    # Show and save
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(depth_color)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(png_save_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved to {png_save_path}")
    plt.close(fig)

