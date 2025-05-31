import os
import torch
from .base_mtl_dataset import BaseMTLDataset, get_pred_name, DatasetMode  # noqa: F401
from .diode_dataset import DIODEDataset
from .hypersim_dataset import HypersimDepthDataset, HypersimNormalDataset, HypersimAlbedoDataset, HypersimShadingDataset
from .kitti_dataset import KITTIDataset
from .vkitti_dataset import VirtualKITTIDataset, VirtualKITTIDepthDataset, \
    VirtualKITTINormalDataset, VirtualKITTISemsegDataset, \
    VirtualKITTIOpticalFlowDataset, VirtualKITTISceneFlowDataset
from .cityscapes_dataset import CityscapesDataset
from .kitti_flow_dataset import KittiFlowDataset
from .flyingthings3d_dataset import FlyingThings3DOpticalFlowDataset, FlyingThings3DSceneFlowDataset
from .mid_intrinsic_dataset import MIDIntrinsicDataset
from tqdm import tqdm


dataset_name_class_dict = {
    "hypersim_normal": HypersimNormalDataset,
    "hypersim_depth": HypersimDepthDataset,
    "hypersim_albedo": HypersimAlbedoDataset,
    "hypersim_shading": HypersimShadingDataset,
    "vkitti": VirtualKITTIDataset,
    "vkitti_depth": VirtualKITTIDepthDataset,
    "vkitti_normal": VirtualKITTINormalDataset,
    "vkitti_semantic": VirtualKITTISemsegDataset,
    "vkitti_optical_flow": VirtualKITTIOpticalFlowDataset,
    "vkitti_scene_flow": VirtualKITTISceneFlowDataset,
    "kitti": KITTIDataset,
    "diode": DIODEDataset,
    "cityscapes": CityscapesDataset,
    "kitti_flow": KittiFlowDataset,
    "flying_things_3D_optical_flow": FlyingThings3DOpticalFlowDataset,
    "flying_things_3D_scene_flow": FlyingThings3DSceneFlowDataset,
    "mid_intrinsic": MIDIntrinsicDataset,
}


def get_dataset(
    cfg_data_split, base_data_dir: str, mode: DatasetMode, **kwargs
) -> BaseMTLDataset:
    if "mixed" == cfg_data_split.name:
        assert DatasetMode.TRAIN == mode, "Only training mode supports mixed datasets."
        dataset_ls = [
            get_dataset(_cfg, base_data_dir, mode, **kwargs)
            for _cfg in cfg_data_split.dataset_list
        ]
        return dataset_ls
    elif cfg_data_split.name in dataset_name_class_dict.keys():
        # import pdb;pdb.set_trace()
        dataset_class = dataset_name_class_dict[cfg_data_split.name]
        dataset = dataset_class(
            mode=mode,
            filename_ls_path=cfg_data_split.filenames,
            dataset_dir=os.path.join(base_data_dir, cfg_data_split.dir),
            **cfg_data_split,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return dataset
