import numpy as np
import torch
from scipy.optimize import nnls


def normalize_scene_flow(scene_flow, type="hw"):
    assert type in ["hw", "norm"], "type must be hw or norm"
    if type == "hw":
        max_x = max(abs(scene_flow[0, :, :].max()), abs(scene_flow[0, :, :].min()))
        max_y = max(abs(scene_flow[1, :, :].max()), abs(scene_flow[1, :, :].min()))
        max_z = max(abs(scene_flow[2, :, :].max()), abs(scene_flow[2, :, :].min()))
        
        scene_flow_norm = scene_flow.clone()
        if max_x > 0:
            scene_flow_norm[0, :, :] = scene_flow[0, :, :] / max_x
        if max_y > 0:
            scene_flow_norm[1, :, :] = scene_flow[1, :, :] / max_y
        if max_z > 0:
            scene_flow_norm[2, :, :] = scene_flow[2, :, :] / max_z
    elif type == "norm":
        max_norm = np.linalg.norm(scene_flow, axis=0).max()
        if max_norm > 0:
            scene_flow_norm = scene_flow / max_norm
        else:
            scene_flow_norm = scene_flow
    return scene_flow_norm

def normalize_optical_flow(optical_flow, type="hw"):
    assert type in ["hw", "norm"], "type must be hw or norm"
    if type == "hw":
        max_x = max(abs(optical_flow[0, :, :].max()), abs(optical_flow[0, :, :].min()))
        max_y = max(abs(optical_flow[1, :, :].max()), abs(optical_flow[1, :, :].min()))
        optical_flow_normalized = optical_flow.clone()
        if max_x > 0:
            optical_flow_normalized[0, :, :] = optical_flow[0, :, :] / max_x
        if max_y > 0:
            optical_flow_normalized[1, :, :] = optical_flow[1, :, :] / max_y
    elif type == "norm":
        max_norm = np.linalg.norm(optical_flow, axis=0).max()
        if max_norm > 0:
            optical_flow_normalized = optical_flow / max_norm
        else:
            optical_flow_normalized = optical_flow
    return optical_flow_normalized



def align_flow_norm_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
):
    gt = gt_arr.squeeze()  # [2, H, W]
    pred = pred_arr.squeeze() # [2, H, W]
    valid_mask = valid_mask_arr.squeeze() # [H, W]

    ori_shape = valid_mask.shape  # input shape
    
    aligned_pred = np.zeros_like(pred_arr)
    scale = 0.0
    shift = 0.0

    gt_norm = np.linalg.norm(gt, axis=0)
    pred_norm = np.linalg.norm(pred, axis=0)

    gt_masked = gt_norm[valid_mask].reshape((-1,))  # NNLS expects 1D array
    pred_masked = pred_norm[valid_mask].reshape((-1, 1))

    # Use non-negative least squares solver
    X, _ = nnls(pred_masked, gt_masked)  # This ensures X is positive
    scale = X[0]

    aligned_pred = pred * scale
    
    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


def align_flow_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    # max_resolution=None,
):
    

    gt = gt_arr.squeeze()  # [2, H, W]
    pred = pred_arr.squeeze() # [2, H, W]
    valid_mask = valid_mask_arr.squeeze() # [H, W]

    ori_shape = valid_mask.shape  # input shape
    
    aligned_pred = np.zeros_like(pred_arr)
    scale = np.zeros((aligned_pred.shape[0],))
    shift = np.zeros((aligned_pred.shape[0],))
    
    
    for i in range(aligned_pred.shape[0]):
        assert (
            gt[i].shape == pred[i].shape == valid_mask.shape
        ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

        gt_masked = gt[i][valid_mask].reshape((-1,))  # NNLS expects 1D array
        pred_masked = pred[i][valid_mask].reshape((-1, 1))

        # Use non-negative least squares solver
        X, _ = nnls(pred_masked, gt_masked)  # This ensures X is positive
        scale[i] = X[0]

        aligned_pred[i] = pred[i] * scale[i]
    
    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


# ******************** disparity space ********************
def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)
