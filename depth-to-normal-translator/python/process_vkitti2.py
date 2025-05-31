from copy import copy
import matplotlib.pyplot as plt
from utils import *
from PIL import Image
from glob import glob
import os
from tqdm import tqdm
import argparse


# version choices: ['d2nt_basic', 'd2nt_v2', 'd2nt_v3']
VERSION = 'd2nt_v3'

def depth2normal(depth_path, cam_fx, cam_fy, u0, v0, normal_save_png_path, normal_save_npy_path):
    depth_img = Image.open(depth_path)
    depth = np.asarray(depth_img) / 100
    # get depth
    h, w = depth.shape
    mask = np.ones_like(depth)
    u_map = np.ones((h, 1)) * np.arange(1, w + 1) - u0  # u-u0
    v_map = np.arange(1, h + 1).reshape(h, 1) * np.ones((1, w)) - v0  # v-v0


    # get depth gradients
    if VERSION == 'd2nt_basic':
        Gu, Gv = get_filter(depth)
    else:
        Gu, Gv = get_DAG_filter(depth)

    # Depth to Normal Translation
    est_nx = Gu * cam_fx
    est_ny = Gv * cam_fy
    est_nz = -(depth + v_map * Gv + u_map * Gu)
    est_normal = cv2.merge((est_nx, est_ny, est_nz))
    est_normal[mask == 0] = 0
    est_normal = - est_normal
    est_normal = vector_normalization(est_normal)


    # MRF-based Normal Refinement
    if VERSION == 'd2nt_v3':
        est_normal = MRF_optim(depth, est_normal)


    # show the computed normal
    # n_vis = visualization_map_creation(est_normal, np.ones_like(mask))
    n_vis = normal_to_rgb(est_normal)
    plt.imsave('normal_vis_vkitti.png', n_vis)
    plt.imsave(normal_save_png_path, n_vis)
    print('Saved normal map to', normal_save_png_path)
    np.save(normal_save_npy_path, est_normal)
    print('Saved normal map to', normal_save_npy_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process depth images to normal maps.')
    parser.add_argument('--output_root', type=str, required=True, help='Root directory for saving processed files')
    args = parser.parse_args()
    scenes = ["Scene01",  "Scene02", "Scene06",  "Scene18",  "Scene20"]

    filename_ls_paths = [
        "../../data_split/vkitti/vkitti_train.txt",
        "../../data_split/vkitti/vkitti_val.txt"
    ]


    for filename_ls_path in filename_ls_paths:
        # Load filenames
        with open(filename_ls_path, "r") as f:
            filenames = [
                s.split() for s in f.readlines()
            ]  # [['rgb.png', 'depth.png'], [], ...]

        cam_fx, cam_fy, u0, v0 =  725.0087, 725.0087, 620.5, 187
        depth_root = os.path.join(args.output_root, "original")
        for filename_line in tqdm(filenames):
            depth_rel_path = filename_line[1]
            depth_path = os.path.join(depth_root, depth_rel_path)

            normal_path = os.path.join(args.output_root, "normal_estimated", depth_rel_path)
            normal_save_npy_path = normal_path.replace('depth', 'normal').replace('png', 'npy')
            normal_save_png_path = normal_path.replace('depth', 'normal')
            output_folder = os.path.dirname(normal_save_npy_path)
            os.makedirs(output_folder, exist_ok=True)

            if os.path.exists(normal_save_npy_path) and os.path.exists(normal_save_png_path):
                print(f"Skipping {normal_save_npy_path} as it already exists")
                continue

            depth2normal(depth_path, cam_fx, cam_fy, u0, v0, normal_save_png_path, normal_save_npy_path)

            # cameras = os.listdir(scene_folder)
            # for camera in cameras:
            #     camera_folder = os.path.join(scene_folder, camera)
            #     list_of_files = glob(camera_folder + '/*.png')
            #     cam_fx, cam_fy, u0, v0 =  725.0087, 725.0087, 620.5, 187
            #     output_folder = os.path.join(args.output_root, "normal_estimated", scene, "clone/frames/normal", camera)
            #     os.makedirs(output_folder, exist_ok=True)
            #     for depth_path in tqdm(list_of_files):
                    # normal_save_npy_path = os.path.join(output_folder, os.path.basename(depth_path).replace('depth', 'normal').replace('png', 'npy'))
                    # normal_save_png_path = os.path.join(output_folder, os.path.basename(depth_path).replace('depth', 'normal'))
                    # depth2normal(depth_path, cam_fx, cam_fy, u0, v0, normal_save_png_path, normal_save_npy_path)






