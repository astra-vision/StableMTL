import torch


class MovingAverageLossWeighter:
    def __init__(self, loss_names,
                 min_weight=0.2, max_weight=5.0,
                 alpha=0.98, epsilon=1e-8):
        """
        Initializes the weighter.

        Args:
            loss_names: List of loss names to balance (e.g., ["a", "b", "c"]).
            alpha: Smoothing factor for EMA (closer to 1 means slower changes).
            epsilon: Small value for numerical stability.
        """
        self.alpha = alpha
        self.epsilon = epsilon
        # Initialize running averages (e.g., with 1.0 or observe first batch)
        self.ema_losses = {loss_name: torch.tensor(1.0) for loss_name in loss_names}
        self.first_update = {loss_name: True for loss_name in loss_names}
        self.min_weight = min_weight
        self.max_weight = max_weight

    def to(self, device):
        """Move internal tensors to the specified device."""
        self.ema_losses = self.ema_losses.to(device)
        return self

    def __call__(self, loss_dict):
        """
        Calculates the combined loss with weights based on EMA of magnitudes.

        Args:
            *losses: The scalar tensor losses (loss_a, loss_b, loss_c, ...).

        Returns:
            Total combined loss, tensor.
        """

        # Ensure EMA tensor is on the same device as losses
        for loss_name in loss_dict:
            if self.ema_losses[loss_name].device != loss_dict[loss_name].device:
                self.ema_losses[loss_name] = self.ema_losses[loss_name].to(loss_dict[loss_name].device)

        # --- Update EMA using detached loss values ---
        for loss_name in loss_dict:
            if self.first_update[loss_name]:
                self.ema_losses[loss_name] = loss_dict[loss_name].detach()
                self.first_update[loss_name] = False
            else:
                self.ema_losses[loss_name] = self.alpha * self.ema_losses[loss_name] + (1 - self.alpha) * loss_dict[loss_name].detach()

        # If any task is in its first update, return uniform weights
        if any(self.first_update.values()):
            return {loss_name: torch.tensor(1.0) for loss_name in loss_dict}

        # --- Calculate weights based on EMA ---
        # Use the same logic as Method 1, but with EMA values
        avg_ema_loss = torch.mean(torch.tensor([self.ema_losses[loss_name] for loss_name in self.ema_losses]))

        weights = {}
        for loss_name in loss_dict:
            # Use clamp(min=epsilon) instead of adding epsilon to handle potential zeros robustly
            weights[loss_name] = avg_ema_loss / self.ema_losses[loss_name].clamp(min=self.epsilon)
            weights[loss_name] = torch.clamp(weights[loss_name], min=self.min_weight, max=self.max_weight)

        return weights


def compute_grad_norm(model):
    # Collect all gradient norms
    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.data.norm(2).item())

    grad_norms = torch.tensor(grad_norms)

    # Compute average norm
    avg_norm = grad_norms.mean().item()
    std_norm = grad_norms.std().item()

    # Compute 90th percentile norm
    # p90_norm = torch.quantile(grad_norms, 0.9).item()

    return avg_norm, std_norm

def get_loss(loss_name, **kwargs):
    if "silog_mse" == loss_name:
        criterion = SILogMSELoss(**kwargs)
    elif "silog_rmse" == loss_name:
        criterion = SILogRMSELoss(**kwargs)
    elif "mse_loss" == loss_name:
        criterion = torch.nn.MSELoss(**kwargs)
    elif "l1_loss" == loss_name:
        criterion = torch.nn.L1Loss(**kwargs)
    elif "l1_loss_with_mask" == loss_name:
        criterion = L1LossWithMask(**kwargs)
    elif "mean_abs_rel" == loss_name:
        criterion = MeanAbsRelLoss()
    else:
        raise NotImplementedError

    return criterion


class L1LossWithMask:
    def __init__(self, batch_reduction=False):
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_pred - depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        loss = torch.sum(torch.abs(diff)) / n
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class MeanAbsRelLoss:
    def __init__(self) -> None:
        # super().__init__()
        pass

    def __call__(self, pred, gt):
        diff = pred - gt
        rel_abs = torch.abs(diff / gt)
        loss = torch.mean(rel_abs, dim=0)
        return loss


class SILogMSELoss:
    def __init__(self, lamb, log_pred=True, batch_reduction=True):
        """Scale Invariant Log MSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogMSELoss, self).__init__()
        self.lamb = lamb
        self.pred_in_log = log_pred
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        log_depth_pred = (
            depth_pred if self.pred_in_log else torch.log(torch.clip(depth_pred, 1e-8))
        )
        log_depth_gt = torch.log(depth_gt)

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)

        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = first_term - second_term
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class SILogRMSELoss:
    def __init__(self, lamb, alpha, log_pred=True):
        """Scale Invariant Log RMSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            alpha:
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogRMSELoss, self).__init__()
        self.lamb = lamb
        self.alpha = alpha
        self.pred_in_log = log_pred

    def __call__(self, depth_pred, depth_gt, valid_mask):
        log_depth_pred = depth_pred if self.pred_in_log else torch.log(depth_pred)
        log_depth_gt = torch.log(depth_gt)
        # borrowed from https://github.com/aliyun/NeWCRFs
        # diff = log_depth_pred[valid_mask] - log_depth_gt[valid_mask]
        # return torch.sqrt((diff ** 2).mean() - self.lamb * (diff.mean() ** 2)) * self.alpha

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)
        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = torch.sqrt(first_term - second_term).mean() * self.alpha
        return loss
