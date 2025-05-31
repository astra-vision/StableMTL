import cv2
import torch
import torchvision
import numpy as np
from .flow_augmentation import flip_image, flip_optical_flow


def flip_normal(normal, valid_mask, flip_mode):
    assert flip_mode in ['lr', 'ud']
    if flip_mode == 'lr':
        normal = np.fliplr(normal).copy()
        if valid_mask is not None:
            valid_mask = np.fliplr(valid_mask).copy()
        normal[:, :, 0] *= -1
    else:
        normal = np.flipud(normal).copy()
        normal[:, :, 1] *= -1
        if valid_mask is not None:
            valid_mask = np.flipud(valid_mask).copy()
    return normal, valid_mask


def random_flip_normal(image1, normal, valid_mask, flip_mode):
    assert flip_mode in ['lr', 'ud']
    assert len(normal.shape) == 3

    if np.random.rand() < 0.5:  # do nothing
        return image1, normal, valid_mask

    # flip images
    new_image1 = flip_image(image1, flip_mode)
    new_normal, new_valid_mask = flip_normal(normal, valid_mask, flip_mode)

    return new_image1, new_normal, new_valid_mask


def random_flip_all_tasks(
    image1, image2=None,
    depth=None, depth_valid_mask=None,
    normal=None, normal_valid_mask=None,
    semantic_rgb_norm=None,
    semantic_valid_mask=None,
    optical_flow=None, optical_flow_valid_mask=None, scene_flow=None,
    shading=None, shading_valid_mask=None, albedo=None,
    flip_mode='lr'
):
    assert flip_mode in ['lr', 'ud']

    ret = {
        'image1': image1,
        'image2': image2,
        'depth': depth,
        'normal': normal,
        'semantic_rgb_norm': semantic_rgb_norm,
        'depth_valid_mask': depth_valid_mask,
        'normal_valid_mask': normal_valid_mask,
        'semantic_valid_mask': semantic_valid_mask,
        'optical_flow': optical_flow,
        'optical_flow_valid_mask': optical_flow_valid_mask,
        'scene_flow': scene_flow,
        'shading': shading,
        'shading_valid_mask': shading_valid_mask,
        'albedo': albedo,
    }

    if np.random.rand() < 0.5:  # do nothing
        return ret

    # flip images
    ret['image1'] = flip_image(image1, flip_mode)
    if image2 is not None:
        ret['image2'] = flip_image(image2, flip_mode)
    else:
        ret['image2'] = None

    if normal is not None:
        ret['normal'], ret['normal_valid_mask'] = flip_normal(normal, normal_valid_mask, flip_mode)
    else:
        ret['normal'], ret['normal_valid_mask'] = None, None

    if depth is not None:
        ret['depth'] = flip_image(depth, flip_mode)
        ret['depth_valid_mask'] = flip_image(depth_valid_mask, flip_mode)
    else:
        ret['depth'], ret['depth_valid_mask'] = None, None

    if semantic_rgb_norm is not None:
        ret['semantic_rgb_norm'] = flip_image(semantic_rgb_norm, flip_mode)
        ret['semantic_valid_mask'] = flip_image(semantic_valid_mask, flip_mode)
    else:
        ret['semantic_rgb_norm'], ret['semantic_valid_mask'] = None, None

    if optical_flow is not None:
        ret['optical_flow'], ret['optical_flow_valid_mask'] = flip_optical_flow(optical_flow, flip_mode, valid_mask=optical_flow_valid_mask)
    else:
        ret['optical_flow'], ret['optical_flow_valid_mask'] = None, None

    if scene_flow is not None:
        ret['scene_flow'], _ = flip_optical_flow(scene_flow, flip_mode, valid_mask=optical_flow_valid_mask)

    if shading is not None:
        ret['shading'] = flip_image(shading, flip_mode)
        ret['shading_valid_mask'] = flip_image(shading_valid_mask, flip_mode)

    if albedo is not None:
        ret['albedo'] = flip_image(albedo, flip_mode)

    return ret


def joint_tasks_augmentation(
    cfgs,
    image1,
    image2=None,
    depth=None,
    normal=None,
    semantic_rgb_norm=None,
    depth_valid_mask=None,
    normal_valid_mask=None,
    semantic_valid_mask=None,
    optical_flow=None,
    optical_flow_valid_mask=None,
    scene_flow=None,
    shading=None,
    shading_valid_mask=None,
    albedo=None,
):
    ret = {
        'image1': image1,
        'image2': image2,
        'depth': depth,
        'normal': normal,
        'semantic_rgb_norm': semantic_rgb_norm,
        'depth_valid_mask': depth_valid_mask,
        'normal_valid_mask': normal_valid_mask,
        'semantic_valid_mask': semantic_valid_mask,
        'optical_flow': optical_flow,
        'optical_flow_valid_mask': optical_flow_valid_mask,
        'scene_flow': scene_flow,
        'shading': shading,
        'shading_valid_mask': shading_valid_mask,
        'albedo': albedo,
    }

    if not cfgs.enabled:
        return ret

    if hasattr(cfgs, 'color_jitter') and cfgs.color_jitter.enabled:
        if image2 is not None:
            ret['image1'], ret['image2'] = color_jitter_two_images(
                ret['image1'], ret['image2'],
                brightness=cfgs.color_jitter.brightness,
                contrast=cfgs.color_jitter.contrast,
                saturation=cfgs.color_jitter.saturation,
                hue=cfgs.color_jitter.hue,
            )
        else:
            ret['image1'] = color_jitter(
                ret['image1'],
                brightness=cfgs.color_jitter.brightness,
                contrast=cfgs.color_jitter.contrast,
                saturation=cfgs.color_jitter.saturation,
                hue=cfgs.color_jitter.hue,
            )
    if hasattr(cfgs, 'random_vertical_flip') and cfgs.random_vertical_flip.enabled:
        ret = random_flip_all_tasks(**ret, flip_mode='ud')


    if hasattr(cfgs, 'random_horizontal_flip') and cfgs.random_horizontal_flip.enabled:
        ret = random_flip_all_tasks(**ret, flip_mode='lr')

    return ret



def joint_normal_augmentation(image1, normal, valid_mask, cfgs):
    if not cfgs.enabled:
        return image1, normal, valid_mask

    if hasattr(cfgs, 'color_jitter') and cfgs.color_jitter.enabled:
        image1 = color_jitter(
            image1,
            brightness=cfgs.color_jitter.brightness,
            contrast=cfgs.color_jitter.contrast,
            saturation=cfgs.color_jitter.saturation,
            hue=cfgs.color_jitter.hue,
        )
    if hasattr(cfgs, 'random_vertical_flip') and cfgs.random_vertical_flip.enabled:
        image1, normal, valid_mask = random_flip_normal(
            image1, normal, valid_mask, flip_mode='ud'
        )

    if hasattr(cfgs, 'random_horizontal_flip') and cfgs.random_horizontal_flip.enabled:
        image1, normal, valid_mask = random_flip_normal(
            image1, normal, valid_mask, flip_mode='lr'
        )

    return image1, normal, valid_mask


def random_flip_depth(image1, depth, depth_valid_mask, flip_mode):
    assert flip_mode in ['lr', 'ud']
    assert len(depth.shape) == 3
    assert len(depth_valid_mask.shape) == 3

    if np.random.rand() < 0.5:  # do nothing
        return image1, depth, depth_valid_mask

    # flip images
    new_image1 = flip_image(image1, flip_mode)
    flip_depth = flip_image(depth, flip_mode)
    flip_depth_valid_mask = flip_image(depth_valid_mask, flip_mode)

    return new_image1, flip_depth, flip_depth_valid_mask


def random_flip_semseg(image1, semseg, semseg_valid_mask, flip_mode):
    assert flip_mode in ['lr', 'ud']
    assert len(semseg.shape) == 3
    assert len(semseg_valid_mask.shape) == 3

    if np.random.rand() < 0.5:  # do nothing
        return image1, semseg, semseg_valid_mask

    # flip images
    new_image1 = flip_image(image1, flip_mode)
    flip_semseg = flip_image(semseg, flip_mode)
    flip_semseg_valid_mask = flip_image(semseg_valid_mask, flip_mode)

    return new_image1, flip_semseg, flip_semseg_valid_mask


def random_flip_albedo_or_shading(image1, albedo_or_shading, albedo_or_shading_valid_mask, flip_mode):
    assert flip_mode in ['lr', 'ud']
    assert len(albedo_or_shading.shape) == 3
    assert len(albedo_or_shading_valid_mask.shape) == 3

    if np.random.rand() < 0.5:  # do nothing
        return image1, albedo_or_shading, albedo_or_shading_valid_mask

    flip_image1 = flip_image(image1, flip_mode)
    flip_albedo_or_shading = flip_image(albedo_or_shading, flip_mode)
    flip_albedo_or_shading_valid_mask = flip_image(albedo_or_shading_valid_mask, flip_mode)

    return flip_image1, flip_albedo_or_shading, flip_albedo_or_shading_valid_mask


def joint_albedo_or_shading_augmentation(image1, albedo_or_shading, albedo_or_shading_valid_mask, cfgs):
    if not cfgs.enabled:
        return image1, albedo_or_shading, albedo_or_shading_valid_mask

    if hasattr(cfgs, 'random_vertical_flip') and cfgs.random_vertical_flip.enabled:
        image1, albedo_or_shading, albedo_or_shading_valid_mask = random_flip_albedo_or_shading(
            image1, albedo_or_shading, albedo_or_shading_valid_mask, flip_mode='ud'
        )

    if hasattr(cfgs, 'random_horizontal_flip') and cfgs.random_horizontal_flip.enabled:
        image1, albedo_or_shading, albedo_or_shading_valid_mask = random_flip_albedo_or_shading(
            image1, albedo_or_shading, albedo_or_shading_valid_mask, flip_mode='lr'
        )

    return image1, albedo_or_shading, albedo_or_shading_valid_mask


def color_jitter_two_images(image1, image2, brightness, contrast, saturation, hue):
    assert image1.shape == image2.shape
    cj_module = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    images = np.concatenate([image1, image2], axis=0)
    images_t = torch.from_numpy(images.transpose([2, 0, 1]).copy())
    images_t = cj_module.forward(images_t / 255.0) * 255.0
    images = images_t.numpy().astype(np.uint8).transpose(1, 2, 0)
    image1, image2 = images[:image1.shape[0]], images[image1.shape[0]:]

    return image1, image2


def color_jitter(image1, brightness, contrast, saturation, hue):
    cj_module = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    images_t = torch.from_numpy(image1.transpose([2, 0, 1]).copy())
    images_t = cj_module.forward(images_t / 255.0) * 255.0
    images = images_t.numpy().astype(np.uint8).transpose(1, 2, 0)

    return images

def joint_depth_augmentation(image1, depth, depth_valid_mask, cfgs):
    if not cfgs.enabled:
        return image1, depth, depth_valid_mask

    if hasattr(cfgs, 'color_jitter') and cfgs.color_jitter.enabled:
        image1 = color_jitter(
            image1,
            brightness=cfgs.color_jitter.brightness,
            contrast=cfgs.color_jitter.contrast,
            saturation=cfgs.color_jitter.saturation,
            hue=cfgs.color_jitter.hue,
        )
    if hasattr(cfgs, 'random_vertical_flip') and cfgs.random_vertical_flip.enabled:
        image1, depth, depth_valid_mask = random_flip_depth(
            image1, depth, depth_valid_mask, flip_mode='ud'
        )

    if hasattr(cfgs, 'random_horizontal_flip') and cfgs.random_horizontal_flip.enabled:
        image1, depth, depth_valid_mask = random_flip_depth(
            image1, depth, depth_valid_mask, flip_mode='lr'
        )

    return image1, depth, depth_valid_mask


def joint_semseg_augmentation(image1, semseg, semseg_valid_mask, cfgs):
    if not cfgs.enabled:
        return image1, semseg, semseg_valid_mask

    if hasattr(cfgs, 'color_jitter') and cfgs.color_jitter.enabled:
        image1 = color_jitter(
            image1,
            brightness=cfgs.color_jitter.brightness,
            contrast=cfgs.color_jitter.contrast,
            saturation=cfgs.color_jitter.saturation,
            hue=cfgs.color_jitter.hue,
        )
    if hasattr(cfgs, 'random_vertical_flip') and cfgs.random_vertical_flip.enabled:
        image1, semseg, semseg_valid_mask = random_flip_semseg(
            image1, semseg, semseg_valid_mask, flip_mode='ud'
        )

    if hasattr(cfgs, 'random_horizontal_flip') and cfgs.random_horizontal_flip.enabled:
        image1, semseg, semseg_valid_mask = random_flip_semseg(
            image1, semseg, semseg_valid_mask, flip_mode='lr'
        )

    return image1, semseg, semseg_valid_mask
