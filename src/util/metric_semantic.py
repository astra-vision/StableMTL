# Author: Bingxin Ke
# Last modified: 2024-02-15


import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import numpy as np


def mIoU(output, target, valid_mask=None):
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

class SemanticMetrics(object):
    def __init__(self, n_classes):
        self.mIoU = 0
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds, valid_masks):
        assert len(label_trues.shape) == 3, "Shape of label_trues must be (bs, H, W)"
        assert len(label_preds.shape) == 3, "Shape of label_preds must be (bs, H, W)"
        assert len(valid_masks.shape) == 3, "Shape of valid_masks must be (bs, H, W)"

            
        for lt, lp, valid_mask in zip(label_trues, label_preds, valid_masks):
            lt = lt[valid_mask]
            lp = lp[valid_mask]
            self.confusion_matrix += self._fast_hist(lt, lp, self.n_classes)

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def _metrics(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        self.mIoU = mean_iu

        return dict(Acc=acc, mIoU=mean_iu), cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def result(self):
        score, class_iou = self._metrics()
        metrics = {f'{k}': v for k, v in score.items()}
        class_m = {f'cls_{k:02d}': v for k, v in class_iou.items()}
        self.iou = class_m
        return {**metrics, **class_m}
