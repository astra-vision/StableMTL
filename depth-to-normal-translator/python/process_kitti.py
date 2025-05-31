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
    est_normal = vector_normalization(est_normal)
    

    # MRF-based Normal Refinement
    if VERSION == 'd2nt_v3':
        est_normal = MRF_optim(depth, est_normal)


    # show the computed normal
    n_vis = visualization_map_creation(est_normal, np.ones_like(mask))
    plt.imsave(normal_save_png_path, n_vis)
    print('Saved normal map to', normal_save_png_path)
    np.save(normal_save_npy_path, est_normal)
    print('Saved normal map to', normal_save_npy_path)
    
    
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process depth images to normal maps.')
    # parser.add_argument('--output_root', type=str, required=True, help='Root directory for saving processed files')
    args = parser.parse_args()
    
    sampled_val_800 = "/lustre/fsn1/projects/rech/kvd/uyl37fq/stablepoop_processed/kitti/kitti_sampled_val_800"
    eigen_test_split_path = "/lustre/fsn1/projects/rech/kvd/uyl37fq/stablepoop_processed/kitti/kitti_eigen_split_test"
    for path in [sampled_val_800, eigen_test_split_path]:
        eigen_seqs = [t for t in os.listdir(path) if "sync" not in t]
        for seq in eigen_seqs:
            calib_path = os.path.join("$DSDIR/KITTI/raw_data/", seq, 'calib_cam_to_cam.txt')
            filedata = read_calib_file(calib_path)
            P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
            K_cam = P_rect_20[0:3, 0:3]
            fx, fy, u0, v0 = K_cam[0, 0], K_cam[1, 1], K_cam[0, 2], K_cam[1, 2]
            syncs = os.listdir(os.path.join(path, seq))
            for sync in syncs:
                estimated_dense_depth = os.path.join(path, sync, "estimated_dense_depth", "groundtruth", "image_02")
                
                depth_files = os.listdir(estimated_dense_depth)
                for depth_file in depth_files:
                    estimated_normal_save_npy_path = os.path.join(estimated_dense_depth, depth_file.replace('dense_depth', 'normal').replace('png', 'npy'))
                    estimated_normal_save_png_path = os.path.join(estimated_dense_depth, depth_file.replace('dense_depth', 'normal'))
                    depth2normal(os.path.join(estimated_dense_depth, depth_file), fx, fy, u0, v0, estimated_normal_save_png_path, estimated_normal_save_npy_path)
                    
    
    
    # scenes = ["Scene01",  "Scene02", "Scene06",  "Scene18",  "Scene20"]
    
    # for scene in scenes:
    #     scene_folder =  "$DSDIR/VirtualKitti2/{}/clone/frames/depth/Camera_0".format(scene)
    #     list_of_files = glob(scene_folder + '/*.png')
    #     cam_fx, cam_fy, u0, v0 =  725.0087, 725.0087, 620.5, 187
    #     output_folder = os.path.join(args.output_root, "{}/normal".format(scene))
    #     os.makedirs(output_folder, exist_ok=True)
    #     for depth_path in tqdm(list_of_files):    
    #         normal_save_npy_path = os.path.join(output_folder, os.path.basename(depth_path).replace('depth', 'normal').replace('png', 'npy'))
    #         normal_save_png_path = os.path.join(output_folder, os.path.basename(depth_path).replace('depth', 'normal'))
    #         depth2normal(depth_path, cam_fx, cam_fy, u0, v0, normal_save_png_path, normal_save_npy_path)


    

    

