# Author: Bingxin Ke
# Last modified: 2024-02-15


import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim


def match_scale(pred, grnd, mask=None):
    if mask is None:
        mask = np.ones(pred.shape[:2]).astype(bool)
    if len(mask.shape) == 3:
        mask = mask.squeeze(-1)

    flat_pred = pred[mask].reshape(-1)
    flat_grnd = grnd[mask].reshape(-1)

    scale, _, _, _ = np.linalg.lstsq(flat_pred.reshape(-1, 1), flat_grnd, rcond=None)

    return scale


class AlbedoAndShadingMetrics(object):
    def __init__(self):
        self.reset()


    def reset(self):
        self.metrics = {
            'rmse': 0.0,
            'ssim': 0.0,
            'lmse': 0.0,
            'count': 0,
        }


    def result(self):
        return {
            'rmse': self.metrics['rmse'] / self.metrics['count'],
            'ssim': self.metrics['ssim'] / self.metrics['count'],
            'lmse': self.metrics['lmse'] / self.metrics['count'],
        }


    def update(self, preds, gts, valid_masks):
        assert len(gts.shape) == 4, f"Shape gts is {gts.shape} != len(4)"
        assert len(preds.shape) == 4, f"Shape preds is {preds.shape} != len(4)"
        assert len(valid_masks.shape) == 4, "Shape of valid_masks must be (bs, H, W)"


        for pred, gt, valid_mask in zip(preds, gts, valid_masks):



            # pred: (3, H, W)
            # gt: (3, H, W)
            # valid_mask: (1, H, W)
            pred = pred.transpose(1, 2, 0) # (H, W, 3)
            gt = gt.transpose(1, 2, 0) # (H, W, 3)
            valid_mask = valid_mask.transpose(1, 2, 0).astype(bool) # (H, W, 1)

            # Apply scale matching with the correct mask shape
            scale = match_scale(pred, gt, valid_mask)
            scaled_pred = (pred * scale).clip(0, 1)
            if False and pred.shape[2] == 1:
                # Draw histograms of pred and gt
                import matplotlib.pyplot as plt
                
                # Create a figure with three subplots side by side
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
                
                # Flatten the arrays for histogram plotting, considering only valid pixels
                valid_pred = pred[valid_mask].flatten()
                valid_gt = gt[valid_mask].flatten()
                valid_scaled_pred = scaled_pred[valid_mask].flatten()
                
                # Plot histogram for prediction
                ax1.hist(valid_pred, bins=50, alpha=0.7, color='blue')
                ax1.set_title('Prediction Histogram')
                ax1.set_xlabel('Pixel Value')
                ax1.set_ylabel('Frequency')
                
                # Plot histogram for ground truth
                ax2.hist(valid_gt, bins=50, alpha=0.7, color='green')
                ax2.set_title('Ground Truth Histogram')
                ax2.set_xlabel('Pixel Value')
                ax2.set_ylabel('Frequency')
                
                scale = scale.item()
                # Plot histogram for scaled prediction
                ax3.hist(valid_scaled_pred, bins=50, alpha=0.7, color='red')
                ax3.set_title(f'Scaled Prediction Histogram (scale={scale:.3f})')
                ax3.set_xlabel('Pixel Value')
                ax3.set_ylabel('Frequency')
                
                # Adjust layout and save the figure
                plt.tight_layout()
                plt.savefig(f'tmp/histogram_comparison_{self.metrics["count"]}.png')
                plt.close(fig)
                breakpoint()

            # if pred.shape[2] == 1:
            #     breakpoint()
            squared_err = (scaled_pred - gt) ** 2
            rmse = np.sqrt(np.sum(squared_err * valid_mask) / np.sum(valid_mask))
            lmse = AlbedoAndShadingMetrics.lmse(gt.squeeze(), scaled_pred.squeeze(), valid_mask.squeeze())
            ssim_value = ssim(scaled_pred, gt, data_range=1.0, channel_axis=2)

            # Update metrics
            self.metrics['rmse'] += rmse
            self.metrics['lmse'] += lmse
            self.metrics['ssim'] += ssim_value
            self.metrics['count'] += 1


    @staticmethod
    def lmse(correct, estimate, mask, window_size=16, window_shift=8):
        """TODO DESCRIPTION

        params:
            correct (TODO): TODO
            estimate (TODO): TODO
            mask (TODO): TODO
            window_size (TODO): TODO
            window_shift (TODO): TODO

        returns:
            (TODO): TODO
        """
        if len(correct.shape) == 2 or correct.shape[-1] == 1:
            lmse = AlbedoAndShadingMetrics.lmse_gray(correct, estimate, mask, window_size, window_shift)
        else:
            lmse = AlbedoAndShadingMetrics.lmse_rgb(correct, estimate, mask, window_size, window_shift)
        return lmse

    @staticmethod
    def lmse_rgb(correct, estimate, mask, window_size, window_shift):
        """Returns the sum of the local sum-squared-errors, where the estimate may be rescaled within
        each local region to minimize the error. The windows are window_size x window_size, and they
        are spaced by window_shift.

        params:
            correct (TODO): TODO
            estimate (TODO): TODO
            mask (TODO): TODO
            window_size (TODO): TODO
            window_shift (TODO): TODO

        returns:
            (float): TODO
        """
        M, N = correct.shape[:2]
        ssq = total = 0.

        for i in range(0, M - window_size + 1, window_shift):
            for j in range(0, N - window_size + 1, window_shift):

                correct_curr = correct[i:i+window_size, j:j+window_size, :]
                estimate_curr = estimate[i:i+window_size, j:j+window_size, :]
                mask_curr = mask[i:i+window_size, j:j+window_size]

                rep_mask = np.concatenate([mask_curr] * 3, 0)
                rep_cor = np.concatenate([
                    correct_curr[:, :, 0],
                    correct_curr[:, :, 1],
                    correct_curr[:, :, 2]],
                    0)
                rep_est = np.concatenate([
                    estimate_curr[:, :, 0],
                    estimate_curr[:, :, 1],
                    estimate_curr[:, :, 2]],
                    0)

                ssq += AlbedoAndShadingMetrics.ssq_error(rep_cor, rep_est, rep_mask)
                # FIX: in the original codebase, this was outdented, which allows
                # for scores greater than 1 (which should not be possible).  On the
                # MIT dataset images, this makes a negligible difference, but on
                # larger images, this can have a significant effect.
                total += np.sum(rep_mask * rep_cor**2)

        assert ~np.isnan(ssq/total)

        return ssq/total


    @staticmethod
    def lmse_gray(correct, estimate, mask, window_size, window_shift):
        """Returns the sum of the local sum-squared-errors, where the estimate may be rescaled within
        each local region to minimize the error. The windows are window_size x window_size, and they
        are spaced by window_shift.

        params:
            correct (TODO): TODO
            estimate (TODO): TODO
            mask (TODO): TODO
            window_size (TODO): TODO
            window_shift (TODO): TODO

        returns:
            (TODO): TODO
        """
        M, N = correct.shape[:2]
        ssq = total = 0.

        for i in range(0, M - window_size + 1, window_shift):
            for j in range(0, N - window_size + 1, window_shift):

                correct_curr = correct[i:i+window_size, j:j+window_size]
                estimate_curr = estimate[i:i+window_size, j:j+window_size]
                mask_curr = mask[i:i+window_size, j:j+window_size]
                ssq += AlbedoAndShadingMetrics.ssq_error(correct_curr, estimate_curr, mask_curr)
                # FIX: in the original codebase, this was outdented, which allows
                # for scores greater than 1 (which should not be possible).  On the
                # MIT dataset images, this makes a negligible difference, but on
                # larger images, this can have a significant effect.
                total += np.sum(mask_curr * correct_curr**2)

        assert ~np.isnan(ssq/total)

        return ssq/total

    @staticmethod
    def ssq_error(correct, estimate, mask):
        """Compute the sum-squared-error for an image, where the estimate is multiplied by a scalar
        which minimizes the error. Sums over all pixels where mask is True. If the inputs are color,
        each color channel can be rescaled independently.

        params:
            correct (TODO): TODO
            estimate (TODO): TODO
            mask (TODO): TODO

        returns:
            (float): TODO
        """
        assert correct.ndim == 2
        if np.sum(estimate**2 * mask) > 1e-5:
            alpha = np.sum(correct * estimate * mask) / np.sum(estimate**2 * mask)
        else:
            alpha = 0.
        return np.sum(mask * (correct - alpha*estimate) ** 2)




