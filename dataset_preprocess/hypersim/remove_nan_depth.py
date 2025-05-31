# Author: Bingxin Ke
# Last modified: 2024-02-19

import argparse
import os

import cv2
import h5py
import numpy as np
import pandas as pd
from hypersim_util import dist_2_depth, tone_map, normal_to_rgb
from tqdm import tqdm
import sklearn
import matplotlib.pyplot as plt
from numpy import isclose, logical_and, where, all, isfinite, sum, logical_not, logical_and, logical_or, logical_xor, logical_not, any
from sklearn import preprocessing

IMG_WIDTH = 1024
IMG_HEIGHT = 768
FOCAL_LENGTH = 886.81



if "__main__" == __name__:
    # Read filenames from the list file
    with open('data_split/hypersim/filename_list_train.txt', 'r') as f:
        depth_files = [line.strip() for line in f.readlines()]
        
    with open('data_split/hypersim/depth_nan.txt', 'r') as f_nan:
        depth_nan_files = [line.strip().split()[0] for line in f_nan.readlines()]
        
        # Extract scene, camera, and frame info from nan files
        nan_info = set()
        for line in depth_nan_files:
            if not line: 
                continue
            parts = line.split('/')
            scene = parts[3]  # e.g. ai_030_005
            camera = "_".join(parts[-2].split('_')[1:3])  # e.g. cam_00
            frame = parts[-1].split('.')[1]  # e.g. 0072
            key = (scene, camera, frame)
            nan_info.add(key)
    
        # Filter out depth files that have nan values
        filtered_depth_files = []
        for line in depth_files:
            parts = line.split()  # Split on whitespace to get all parts
            depth_path = parts[1]  # depth path is second part
            # Extract info from depth path
            scene = depth_path.split('/')[0]  # e.g. ai_055_010
            
            # Parse from depth_plane_cam_01_fr0089.png format
            filename = depth_path.split('/')[1]  # depth_plane_cam_01_fr0089.png
            camera = filename.split('depth_plane_')[1].split('_fr')[0]  # cam_01
            frame = filename.split('_fr')[1].split('.')[0]  # 0089
            
            if (scene, camera, frame) not in nan_info:
                filtered_depth_files.append(line)
        # Add debug prints
        print(f"Original number of files: {len(depth_files)}")
        print(f"Number of nan entries: {len(nan_info)}")
        print(f"Number of filtered files: {len(filtered_depth_files)}")
        
        # Print a few examples of filtered out files for verification
        filtered_out = set(depth_files) - set(filtered_depth_files)
        print("\nExample filtered out files (first 5):")
        for file in list(filtered_out)[:5]:
            print(file)
            

    with open('data_split/hypersim/filename_list_train_no_nandepth.txt', 'w') as f:
        for line in filtered_depth_files:
            f.write(line + '\n')

