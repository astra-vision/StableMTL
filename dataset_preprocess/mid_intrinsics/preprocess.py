# NOTE: taken from https://github.dev/MCG-NJU/CamLiFlow

import os
import cv2
import shutil
import logging
import argparse
import torch.utils.data
import numpy as np
from tqdm import tqdm
import glob
from PIL import Image
from pylab import count_nonzero, clip, np
import random
import OpenEXR
import imageio
import Imath
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"



parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=False,
                    # default="/lustre/fsstor/projects/rech/kvd/uyl37fq/mid_intrinsics",
                    default="/scratch/_projets_/astra/stablemtl/data/mid_intrinsics",
                    help='Path to the FlyingThings3D subset')
parser.add_argument('--output_dir', required=False,
                    # default='/lustre/fsn1/projects/rech/kvd/uyl37fq/stablepoop_processed/mid_intrinsics'
                    default="/scratch/_projets_/astra/stablemtl/preprocessed/mid_intrinsics"
                    )
args = parser.parse_args()


def tone_map(rgb):

    gamma = 1.0 / 2.2  # standard gamma correction exponent
    inv_gamma = 1.0 / gamma
    percentile = (
        90  # we want this percentile brightness value in the unmodified image...
    )
    brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling

    valid_mask = np.ones_like(rgb[:, :, 0]).astype(bool)

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


def get_tonemap_scale(rgb_color, p=90):
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
    # brightness_valid = brightness[valid_mask]

    # if the kth percentile brightness value in the unmodified image is less than this,
    # set the scale to 0.0 to avoid divide-by-zero
    eps = 0.0001
    brightness_nth_percentile_current = np.percentile(brightness, p)

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

def readexr(path):
    """EXR read helper. Requires OpenEXR."""
    import OpenEXR

    fh = OpenEXR.InputFile(path)
    dw = fh.header()['dataWindow']
    w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    rgb = [np.ndarray([h,w], dtype="float32", buffer=fh.channel(c)) for c in ['R', 'G', 'B']]
    ret = np.zeros([h, w, 3], dtype='float32')
    for i in [0,1,2]:
        ret[:,:,i] = rgb[i]
    return ret


def writejpg(I, path):
  """JPG write helper. Requires PIL."""

  from PIL import Image
  im = Image.fromarray(I)
  im.save(path, "JPEG", quality=95)


class Preprocessor(torch.utils.data.Dataset):
    def __init__(self, input_dir, output_dir, split):
        super(Preprocessor, self).__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.split = split
        self.scenes = glob.glob(os.path.join(input_dir, split, '*'))
        self.rgb_files = []
        for scene in self.scenes:
            files = glob.glob(os.path.join(scene, '*mip2.exr'))
            self.rgb_files.extend(files)

        self.split_dir = os.path.join(self.output_dir, self.split)
        os.makedirs(self.split_dir, exist_ok=True)


    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, i):
        np.random.seed(0)
        rgb_file = self.rgb_files[i]

        rgb_name = os.path.basename(rgb_file).split('.')[0]
        scene_dir = os.path.dirname(rgb_file)
        scene_name = os.path.basename(scene_dir)



        rgb_hdr = readexr(rgb_file)
        tm_scale = get_tonemap_scale(rgb_hdr)
        rgb = (tm_scale * rgb_hdr).clip(0, 1)
        albedo_file = os.path.join(scene_dir, 'albedo.exr')
        albedo = cv2.imread(albedo_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
        
        shading = (rgb / albedo).clip(0, 1)


        
        rgb = (rgb * 255).astype(np.uint8)
        rgb = Image.fromarray(rgb)
        rgb_save_path = os.path.join(self.split_dir, f"{scene_name}_{rgb_name}_scaled_only.jpg")
        rgb.save(rgb_save_path)
        print(f">> Saved rgb to {rgb_save_path}")

        rgb = tone_map(rgb_hdr)
        rgb = (rgb * 255).astype(np.uint8)
        rgb = Image.fromarray(rgb)
        rgb_save_path = os.path.join(self.split_dir, f"{scene_name}_{rgb_name}.jpg")
        rgb.save(rgb_save_path)
        print(f">> Saved rgb to {rgb_save_path}")

        albedo = albedo.clip(0, 1)
        albedo = (albedo * 255.0).astype(np.uint8)
        albedo = Image.fromarray(albedo)
        albedo_save_path = os.path.join(self.split_dir, f"{scene_name}_{rgb_name}_albedo.jpg")
        albedo.save(albedo_save_path)
        print(f">> Saved albedo to {albedo_save_path}")


        shading = (shading * 255).astype(np.uint8)
        shading_save_path = os.path.join(self.split_dir, f"{scene_name}_{rgb_name}_shading.jpg")
        shading = Image.fromarray(shading)
        shading.save(shading_save_path)
        print(f">> Saved shading to {shading_save_path}")


        return rgb_save_path


def main():
    for split_idx, split in enumerate(['test']):

        preprocessor = Preprocessor(
            args.input_dir,
            args.output_dir,
            split
        )
        preprocessor[0]
        preprocessor = torch.utils.data.DataLoader(dataset=preprocessor, num_workers=8)

        path_list = []
        for flow_path in tqdm(preprocessor):
            abs_path = flow_path[0]
            rel_path = abs_path.replace(f"{args.output_dir}/", '')
            path_list.append(rel_path)

        split_txt = f"data_split/mid_intrinsics/{split}.txt"
        os.makedirs(os.path.dirname(split_txt), exist_ok=True)
        with open(split_txt, 'w') as f:
            for path in path_list:
                f.write(path + '\n')
        # Create lite version with randomly sampled subset
        num_samples = 300
        sampled_paths = random.sample(path_list, min(num_samples, len(path_list)))

        split_lite_txt = f"data_split/mid_intrinsics/{split}_lite_300.txt"
        with open(split_lite_txt, 'w') as f:
            for path in sampled_paths:
                f.write(path + '\n')
        print(f">> Saved {len(sampled_paths)} samples to {split_lite_txt}")

        # Create visualization version with 20 randomly sampled images
        num_vis_samples = 20
        vis_paths = random.sample(path_list, min(num_vis_samples, len(path_list)))

        split_vis_txt = f"data_split/mid_intrinsics/{split}_vis_20.txt"
        with open(split_vis_txt, 'w') as f:
            for path in vis_paths:
                f.write(path + '\n')
        print(f">> Saved {len(vis_paths)} samples to {split_vis_txt}")

if __name__ == '__main__':
    main()
    logging.info('All done.')
