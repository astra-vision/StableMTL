# Author: Bingxin Ke
# Last modified: 2024-04-18

import torch
import logging


def get_optical_flow_normalizer(cfg_normalizer):
    if cfg_normalizer is None:

        def identical(x):
            return x

        depth_transform = identical

    elif "scale_shift_optical_flow" == cfg_normalizer.type:
        depth_transform = ScaleShiftOpticalFlowNormalizer(
            norm_min=cfg_normalizer.norm_min,
            norm_max=cfg_normalizer.norm_max,
            min_max_quantile=cfg_normalizer.min_max_quantile,
            clip=cfg_normalizer.clip,
        )
    else:
        raise NotImplementedError
    return depth_transform


class OpticalFlowNormalizerBase:
    def __init__(
        self,
        norm_min=-1.0,
        norm_max=1.0,
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        raise NotImplementedError

    def __call__(self, depth, valid_mask=None, clip=None):
        raise NotImplementedError

    def denormalize(self, depth_norm, **kwargs):
        # For metric depth: convert prediction back to metric depth
        # For relative depth: convert prediction to [0, 1]
        raise NotImplementedError


class ScaleShiftOpticalFlowNormalizer(OpticalFlowNormalizerBase):
    """
    Use max and mix flow value to linearly normalize optical_flow,
        i.e. f' = f * s + t,
        where min flow is mapped to `norm_min`, and max flow is mapped to `norm_max`
    """


    def __init__(
        self, norm_min=-1.0, norm_max=1.0, min_max_quantile=0.00, clip=True
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_range = self.norm_max - self.norm_min
        self.min_quantile = min_max_quantile
        self.max_quantile = 1.0 - self.min_quantile
        self.clip = clip

    def __call__(self, optical_flow, valid_mask=None, clip=None):
        """
        optical_flow: [2, H, W]
        valid_mask: [1, H, W]
        """
        _, H, W = optical_flow.shape
        if valid_mask is None:
            valid_mask = torch.ones((1, H, W), dtype=torch.bool)
        
        valid_mask = valid_mask.squeeze(0)
        
        
        clip = clip if clip is not None else self.clip

        _min_x, _max_x = torch.quantile(
            optical_flow[0][valid_mask],
            torch.tensor([self.min_quantile, self.max_quantile]),
        )
        _min_y, _max_y = torch.quantile(
            optical_flow[1][valid_mask],
            torch.tensor([self.min_quantile, self.max_quantile]),
        )
        max_x = torch.max(torch.abs(_min_x), torch.abs(_max_x))
        max_y = torch.max(torch.abs(_min_y), torch.abs(_max_y))
        
        optical_flow_norm_linear = torch.zeros_like(optical_flow)
        
        # NOTE: this avoids error when flipping the sign of optical_flow
        optical_flow_norm_linear[0] = optical_flow[0] / max_x
        optical_flow_norm_linear[1] = optical_flow[1] / max_y
        
        # scale and shift x
        # optical_flow_norm_linear[0] = (
        #     optical_flow[0] - _min_x
        # ) / (_max_x - _min_x) * self.norm_range + self.norm_min
        
        # # scale and shift y
        # optical_flow_norm_linear[1] = (
        #     optical_flow[1] - _min_y
        # ) / (_max_y - _min_y) * self.norm_range + self.norm_min
        
        if clip:
            optical_flow_norm_linear = torch.clip(
                optical_flow_norm_linear, self.norm_min, self.norm_max
            )

        return optical_flow_norm_linear
    

    def scale_back(self, depth_norm):
        # scale to [0, 1]
        depth_linear = (depth_norm - self.norm_min) / self.norm_range
        return depth_linear

    def denormalize(self, depth_norm, **kwargs):
        logging.warning(f"{self.__class__} is not revertible without GT")
        return self.scale_back(depth_norm=depth_norm)
