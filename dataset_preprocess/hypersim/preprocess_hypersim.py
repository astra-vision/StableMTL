# Author: Bingxin Ke
# Last modified: 2024-02-19

import argparse
import os

import cv2
import h5py
import numpy as np
import pandas as pd
from hypersim_util import dist_2_depth, tone_map, normal_to_rgb, get_tonemap_scale
from tqdm import tqdm
import sklearn
import matplotlib.pyplot as plt
from numpy import isclose, logical_and, where, all, isfinite, sum, logical_not, logical_and, logical_or, logical_xor, logical_not, any
from sklearn import preprocessing


IMG_WIDTH = 1024
IMG_HEIGHT = 768
FOCAL_LENGTH = 886.81

filtered_out_list = ['ai_004_009/rgb_cam_01_fr0000.png',
                     'ai_008_001/rgb_cam_01_fr0000.png',
                     'ai_008_001/rgb_cam_02_fr0000.png',
                     'ai_011_005/rgb_cam_01_fr0000.png',
                     'ai_016_009/rgb_cam_00_fr0000.png',
                     'ai_052_002/rgb_cam_01_fr0021.png']

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_csv",
        type=str,
        default="/lustre/fswork/projects/rech/trg/uyl37fq/code/stablepoop/data_split/hypersim/metadata_images_split_scene_v1.csv",
    )
    parser.add_argument("--dataset_dir", type=str, default="$DSDIR/Hypersim/evermotion_dataset/scenes")
    parser.add_argument("--output_dir", type=str, default="/lustre/fsn1/projects/rech/kvd/uyl37fq/preprocessed/hypersim")
    parser.add_argument("--process_id", type=int, default=0)
    parser.add_argument("--n_processes", type=int, default=1)

    args = parser.parse_args()

    split_csv = args.split_csv
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # %%
    raw_meta_df = pd.read_csv(split_csv)
    meta_df = raw_meta_df[raw_meta_df.included_in_public_release].copy()


    # %%
    # "val", "test"
    for split in ["train"]:
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        split_meta_df = meta_df[meta_df.split_partition_name == split].copy()
        split_meta_df["rgb_path"] = None
        split_meta_df["rgb_mean"] = np.nan
        split_meta_df["rgb_std"] = np.nan
        split_meta_df["rgb_min"] = np.nan
        split_meta_df["rgb_max"] = np.nan
        split_meta_df["depth_path"] = None
        split_meta_df["depth_mean"] = np.nan
        split_meta_df["depth_std"] = np.nan
        split_meta_df["depth_min"] = np.nan
        split_meta_df["depth_max"] = np.nan
        split_meta_df["invalid_ratio"] = np.nan

        for i, row in tqdm(split_meta_df.iloc[args.process_id::args.n_processes].iterrows(),
                           total=len(split_meta_df)//args.n_processes):
            scene_path = row.scene_name
            if not os.path.exists(os.path.join(split_output_dir, row.scene_name)):
                os.makedirs(os.path.join(split_output_dir, row.scene_name))


            depth_name = f"depth_plane_{row.camera_name}_fr{row.frame_id:04d}.png"
            reflectance_name = f"reflectance_{row.camera_name}_fr{row.frame_id:04d}.png"
            rgb_name = f"rgb_{row.camera_name}_fr{row.frame_id:04d}.png"

            norm_png_name = f"normal_cam_{row.camera_name}_fr{row.frame_id:04d}.png"
            norm_npy_name = f"normal_cam_{row.camera_name}_fr{row.frame_id:04d}.npy"
            norm_valid_mask_name = f"normal_valid_mask_{row.camera_name}_fr{row.frame_id:04d}.npy"
            shading_name = f"shading_{row.camera_name}_fr{row.frame_id:04d}.png"
            albedo_name = f"albedo_{row.camera_name}_fr{row.frame_id:04d}.png"

            save_depth_path = os.path.join(scene_path, depth_name)
            save_rgb_path = os.path.join(scene_path, rgb_name)
            save_reflectance_path = os.path.join(scene_path, reflectance_name)
            save_norm_png_path = os.path.join(scene_path, norm_png_name)
            save_norm_npy_path = os.path.join(scene_path, norm_npy_name)
            save_normal_valid_mask_path = os.path.join(scene_path, norm_valid_mask_name)
            save_shading_path = os.path.join(scene_path, shading_name)
            save_albedo_path = os.path.join(scene_path, albedo_name)

            if save_rgb_path in filtered_out_list:
                continue


            if not (os.path.exists(os.path.join(split_output_dir, save_shading_path)) \
                and os.path.exists(os.path.join(split_output_dir, save_albedo_path))):
                render_entity_id_path = os.path.join(
                    row.scene_name,
                    "images",
                    f"scene_{row.camera_name}_geometry_hdf5",
                    f"frame.{row.frame_id:04d}.render_entity_id.hdf5",
                )

                reflectance_path = os.path.join(
                    row.scene_name,
                    "images",
                    f"scene_{row.camera_name}_final_hdf5",
                    f"frame.{row.frame_id:04d}.diffuse_reflectance.hdf5",
                )

                illumination_path = os.path.join(
                    row.scene_name,
                    "images",
                    f"scene_{row.camera_name}_final_hdf5",
                    f"frame.{row.frame_id:04d}.diffuse_illumination.hdf5",
                )

                rgb_path = os.path.join(
                    row.scene_name,
                    "images",
                    f"scene_{row.camera_name}_final_hdf5",
                    f"frame.{row.frame_id:04d}.color.hdf5",
                )

                with h5py.File(os.path.join(dataset_dir, rgb_path), "r") as f:
                    rgb = np.array(f["dataset"]).astype(float)
                with h5py.File(os.path.join(dataset_dir, reflectance_path), "r") as f:
                    reflectance = np.array(f["dataset"]).astype(float)
                with h5py.File(os.path.join(dataset_dir, render_entity_id_path), "r") as f:
                    render_entity_id = np.array(f["dataset"]).astype(int)

                valid_mask = render_entity_id != -1
                if not any(valid_mask):
                    print(f"Skipping {save_shading_path} as it has no valid pixels")
                    continue
                tonemap_scale = get_tonemap_scale(rgb, valid_mask=valid_mask)
                shading = (rgb / (reflectance + 1e-6))
                albedo = (tonemap_scale * reflectance).clip(0, 1)
                shading = shading.clip(0, 1)

                cv2.imwrite(
                    os.path.join(split_output_dir, save_shading_path),
                    cv2.cvtColor((shading * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                )
                print(f"Saved shading to {save_shading_path}", flush=True)

                cv2.imwrite(
                    os.path.join(split_output_dir, save_albedo_path),
                    cv2.cvtColor((albedo * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                )
                print(f"Saved albedo to {save_albedo_path}", flush=True)
            else:
                print(f"Skipping {save_shading_path} as it already exists")

            if not (os.path.exists(os.path.join(split_output_dir, save_depth_path)) \
                and os.path.exists(os.path.join(split_output_dir, save_norm_png_path)) \
                and os.path.exists(os.path.join(split_output_dir, save_normal_valid_mask_path)) \
                and os.path.exists(os.path.join(split_output_dir, save_norm_npy_path)) \
                and os.path.exists(os.path.join(split_output_dir, save_rgb_path))):

                # Load data
                rgb_path = os.path.join(
                    row.scene_name,
                    "images",
                    f"scene_{row.camera_name}_final_hdf5",
                    f"frame.{row.frame_id:04d}.color.hdf5",
                )

                dist_path = os.path.join(
                    row.scene_name,
                    "images",
                    f"scene_{row.camera_name}_geometry_hdf5",
                    f"frame.{row.frame_id:04d}.depth_meters.hdf5",
                )

                normal_cam_path = os.path.join(
                    row.scene_name,
                    "images",
                    f"scene_{row.camera_name}_geometry_hdf5",
                    f"frame.{row.frame_id:04d}.normal_cam.hdf5",
                )

                normal_world_path = os.path.join(
                    row.scene_name,
                    "images",
                    f"scene_{row.camera_name}_geometry_hdf5",
                    f"frame.{row.frame_id:04d}.normal_world.hdf5",
                )

                render_entity_id_path = os.path.join(
                    row.scene_name,
                    "images",
                    f"scene_{row.camera_name}_geometry_hdf5",
                    f"frame.{row.frame_id:04d}.render_entity_id.hdf5",
                )

                position_path = os.path.join(
                    row.scene_name,
                    "images",
                    f"scene_{row.camera_name}_geometry_hdf5",
                    f"frame.{row.frame_id:04d}.position.hdf5",
                )

                camera_keyframe_frame_indices_hdf5_file = os.path.join(
                    row.scene_name,
                    "_detail",
                    row.camera_name,
                    "camera_keyframe_frame_indices.hdf5",
                )
                camera_keyframe_positions_hdf5_file = os.path.join(
                    row.scene_name,
                    "_detail",
                    row.camera_name,
                    "camera_keyframe_positions.hdf5",
                )



                assert os.path.exists(os.path.join(dataset_dir, rgb_path))
                assert os.path.exists(os.path.join(dataset_dir, dist_path))

                with h5py.File(os.path.join(dataset_dir, rgb_path), "r") as f:
                    rgb = np.array(f["dataset"]).astype(float)

                with h5py.File(os.path.join(dataset_dir, dist_path), "r") as f:
                    dist_from_center = np.array(f["dataset"]).astype(float)
                with h5py.File(os.path.join(dataset_dir, render_entity_id_path), "r") as f:
                    render_entity_id = np.array(f["dataset"]).astype(int)
                with h5py.File(os.path.join(dataset_dir, normal_cam_path), "r") as f:
                    normal_cam = np.array(f["dataset"]).astype(float)
                with h5py.File(os.path.join(dataset_dir, normal_world_path), "r") as f:
                    normal_world = np.array(f["dataset"]).astype(float)
                with h5py.File(os.path.join(dataset_dir, position_path), "r") as f:
                    position = np.array(f["dataset"]).astype(float)
                with h5py.File(os.path.join(dataset_dir, camera_keyframe_positions_hdf5_file), "r") as f:
                    camera_positions = np.array(f["dataset"]).astype(float)

                # Apply tone mapping to the RGB image and convert to uint8 format
                rgb_color_tm = tone_map(rgb, render_entity_id)
                rgb_int = (rgb_color_tm * 255).astype(np.uint8)  # [H, W, RGB]

            
                # Distance -> depth
                plane_depth = dist_2_depth(
                    IMG_WIDTH, IMG_HEIGHT, FOCAL_LENGTH, dist_from_center
                )
                valid_mask = render_entity_id != -1
                valid_mask_ = valid_mask.copy()

                # Record invalid ratio
                invalid_ratio = (np.prod(valid_mask.shape) - valid_mask.sum()) / np.prod(
                    valid_mask.shape
                )
                plane_depth[~valid_mask] = 0

                if np.isnan(plane_depth).all():
                    print(f"Skipping {save_depth_path} as all values are nan")
                    continue

                count_nan = np.sum(np.isnan(plane_depth))
                if count_nan > 0:
                    print("Before: number of nan in plane_depth", count_nan)
                    plane_depth = np.nan_to_num(plane_depth, nan=0)
                    print("After number of nan in plane_depth", np.sum(np.isnan(plane_depth)))

                # save RGB
                cv2.imwrite(
                    os.path.join(split_output_dir, save_rgb_path),
                    cv2.cvtColor(rgb_int, cv2.COLOR_RGB2BGR),
                )



                plane_depth *= 1000.0
                plane_depth = plane_depth.astype(np.uint16)
                cv2.imwrite(os.path.join(split_output_dir, save_depth_path), plane_depth)
                


                infinite_vals_mask = logical_not(all(isfinite(position), axis=2))
                if any(logical_and(valid_mask, infinite_vals_mask)):
                    warning_pixels_mask = logical_and(valid_mask, infinite_vals_mask)
                    warning_pixels_y, warning_pixels_x = where(warning_pixels_mask)
                    print("[HYPERSIM: DATASET_GENERATE_IMAGE_STATISTICS] WARNING: NON-FINITE VALUE AT VALID PIXEL IN " + position_path + " (num_pixels=" + str(warning_pixels_y.shape[0]) + "; see pixel y=" + str(warning_pixels_y[0]) + ", x=" + str(warning_pixels_x[0]) + ")")
                valid_mask_[infinite_vals_mask] = False

                zero_normals_mask = all(isclose(normal_cam, 0.0), axis=2)
                if any(logical_and(valid_mask, zero_normals_mask)):
                    warning_pixels_mask = logical_and(valid_mask, zero_normals_mask)
                    warning_pixels_y, warning_pixels_x = where(warning_pixels_mask)
                    print("[HYPERSIM: DATASET_GENERATE_IMAGE_STATISTICS] WARNING: ZERO NORMALS AT VALID PIXEL IN " + normal_cam_path + " (num_pixels=" + str(warning_pixels_y.shape[0]) + "; see pixel y=" + str(warning_pixels_y[0]) + ", x=" + str(warning_pixels_x[0]) + ")")
                valid_mask_[zero_normals_mask] = False

                zero_normals_mask = all(isclose(normal_world, 0.0), axis=2)
                if any(logical_and(valid_mask, zero_normals_mask)):
                    warning_pixels_mask = logical_and(valid_mask, zero_normals_mask)
                    warning_pixels_y, warning_pixels_x = where(warning_pixels_mask)
                    print("[HYPERSIM: DATASET_GENERATE_IMAGE_STATISTICS] WARNING: ZERO NORMALS AT VALID PIXEL IN " + normal_world_path + " (num_pixels=" + str(warning_pixels_y.shape[0]) + "; see pixel y=" + str(warning_pixels_y[0]) + ", x=" + str(warning_pixels_x[0]) + ")")

                valid_mask_[zero_normals_mask] = False

                valid_mask   = valid_mask_
                invalid_mask = logical_not(valid_mask)

                valid_mask_1d   = valid_mask.reshape(-1)
                invalid_mask_1d = invalid_mask.reshape(-1)

                if not any(valid_mask_1d):
                    print("[HYPERSIM: DATASET_GENERATE_IMAGE_STATISTICS] WARNING: NO VALID PIXELS, SKIPPING...")
                    continue

                normal_cam_1d_ = normal_cam.reshape(-1,3)
                normal_cam_1d_[invalid_mask_1d] = -987654321.0
                normal_cam_1d_ = preprocessing.normalize(normal_cam_1d_)
                normal_cam     = normal_cam_1d_.reshape(normal_cam.shape)

                normal_world_1d_ = normal_world.reshape(-1,3)
                normal_world_1d_[invalid_mask_1d] = -987654321.0
                normal_world_1d_ = preprocessing.normalize(normal_world_1d_)
                normal_world     = normal_world_1d_.reshape(normal_world.shape)



                # orient normals towards the camera - should do this when generating the HDF5 data?
                position_1d_     = position.reshape(-1,3)
                normal_world_1d_ = normal_world.reshape(-1,3)

                position_1d_[invalid_mask_1d]     = -987654321.0
                normal_world_1d_[invalid_mask_1d] = -987654321.0


                assert all(isfinite(position_1d_))
                camera_position = camera_positions[row.frame_id]
                surface_to_cam_world_normalized_1d_ = sklearn.preprocessing.normalize(camera_position - position_1d_)
                n_dot_v_1d_                         = sum(normal_world_1d_*surface_to_cam_world_normalized_1d_, axis=1)
                normal_back_facing_mask_1d_         = logical_and(valid_mask_1d, n_dot_v_1d_ < 0)
                normal_back_facing_mask             = normal_back_facing_mask_1d_.reshape(normal_world.shape[0], normal_world.shape[1])
                normal_back_facing_mask_1d          = normal_back_facing_mask_1d_.reshape(-1)

                normal_cam_ = normal_cam.copy()
                normal_cam_[normal_back_facing_mask] = -normal_cam_[normal_back_facing_mask]
                normal_cam_[:, :, 0][valid_mask] = -normal_cam_[:, :, 0][valid_mask]
                normal_cam_1d                        = normal_cam_.reshape(-1,3)[valid_mask_1d]
                assert np.allclose(np.linalg.norm(normal_cam_1d, axis=1), 1.0)
                normal_cam = normal_cam_




                normal_vis = normal_to_rgb(normal_cam)


                plt.imsave(os.path.join(split_output_dir, save_norm_png_path), normal_vis)
                np.save(os.path.join(split_output_dir, save_norm_npy_path), normal_cam)
                np.save(os.path.join(split_output_dir, save_normal_valid_mask_path), valid_mask)
                
                print(f"Saved depth to {save_depth_path}")
                print(f"Saved normal to {save_norm_npy_path}")
                print(f"Saved RGB to {save_rgb_path}")

                # Meta data

                split_meta_df.at[i, "rgb_mean"] = np.mean(rgb_int)
                split_meta_df.at[i, "rgb_std"] = np.std(rgb_int)
                split_meta_df.at[i, "rgb_min"] = np.min(rgb_int)
                split_meta_df.at[i, "rgb_max"] = np.max(rgb_int)


                restored_depth = plane_depth / 1000.0
                split_meta_df.at[i, "depth_mean"] = np.mean(restored_depth)
                split_meta_df.at[i, "depth_std"] = np.std(restored_depth)
                split_meta_df.at[i, "depth_min"] = np.min(restored_depth)
                split_meta_df.at[i, "depth_max"] = np.max(restored_depth)

                split_meta_df.at[i, "invalid_ratio"] = invalid_ratio
            else:
                print(f"Skipping {save_depth_path} as it already exists")
                print(f"Skipping {save_norm_npy_path} as it already exists")
                print(f"Skipping {save_rgb_path} as it already exists")


    print("Preprocess finished")
