# NOTE: taken from https://github.dev/MCG-NJU/CamLiFlow

import os
import cv2
import shutil
import logging
import argparse
import torch.utils.data
import numpy as np
from tqdm import tqdm
from utils import init_logging, load_fpm, load_flow, disp2pc, save_flow_png


'''
Download the "FlyingThings3D subset" from:
https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
You need to download the following part:
* RGB images (cleanpass)
* Disparity
* Disparity change
* optical_flow
* Flow occlusions
Uncompress them and organize the structure of directory as follows:
/mnt/data/flyingthings3d_subset
├── train
│   ├── disparity
│   ├── disparity_change
│   ├── disparity_occlusions
│   ├── flow
│   ├── flow_occlusions
│   └── image_clean
└── val
    ├── disparity
    ├── disparity_change
    ├── disparity_occlusions
    ├── flow
    ├── flow_occlusions
    └── image_clean
Then preprocess the data:
python preprocess_flyingthings3d_subset.py --input_dir /mnt/data/flyingthings3d_subset
'''

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=False, 
                    default="/lustre/fsn1/projects/rech/kvd/uyl37fq/data/FlyingThings3D_subset",
                    # default="/scratch/_projets_/astra/stablemtl/data/FlyingThings3D/FlyingThings3D_subset",
                    help='Path to the FlyingThings3D subset')
parser.add_argument('--output_dir', required=False, 
                    default='/lustre/fsn1/projects/rech/kvd/uyl37fq/stablepoop_processed/FlyingThings3D_preprocessed'
                    # default="/scratch/_projets_/astra/stablemtl/preprocessed/FlyingThings3D"
                    )
parser.add_argument('--n_points', required=False, default=[32768, 8192])
parser.add_argument('--max_depth', required=False, default=35.0)
parser.add_argument('--remove_occluded_points', action='store_true')
args = parser.parse_args()


class Preprocessor(torch.utils.data.Dataset):
    def __init__(self, input_dir, output_dir, split, n_points, max_depth, remove_occluded_points):
        super(Preprocessor, self).__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.split = split
        self.n_points = n_points
        self.max_depth = max_depth
        self.remove_occluded_points = remove_occluded_points

        self.indices = []
        for filename in os.listdir(os.path.join(input_dir, split, 'flow', 'left', 'into_future')):
            index = int(filename.split('.')[0])
            self.indices.append(index)
        
        

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        np.random.seed(0)

        index1 = self.indices[i]
        index2 = index1 + 1

        # camera intrinsics
        baseline, f, cx, cy = 1.0, 1050.0, 479.5, 269.5

        # load data
        disp1 = -load_fpm(os.path.join(
            self.input_dir, self.split, 'disparity', 'left', '%07d.pfm' % index1
        ))
        disp2 = -load_fpm(os.path.join(
            self.input_dir, self.split, 'disparity', 'left', '%07d.pfm' % index2
        ))
        disp1_change = -load_fpm(os.path.join(
            self.input_dir, self.split, 'disparity_change', 'left', 'into_future', '%07d.pfm' % index1
        ))
        flow_2d = load_flow(os.path.join(
            self.input_dir, self.split, 'flow', 'left', 'into_future', '%07d.flo' % index1
        ))
        occ_mask_2d = cv2.imread(os.path.join(
            self.input_dir, self.split, 'flow_occlusions', 'left', 'into_future', '%07d.png' % index1
        ))
        occ_mask_2d = occ_mask_2d[..., 0] > 1

        if self.remove_occluded_points:
            pc1 = disp2pc(disp1, baseline, f, cx, cy)
            pc2 = disp2pc(disp1 + disp1_change, baseline, f, cx, cy, flow_2d)

            # apply non-occlusion mask
            noc_mask_2d = np.logical_not(occ_mask_2d)
            pc1, pc2 = pc1[noc_mask_2d], pc2[noc_mask_2d]

            # apply depth mask
            mask = np.logical_and(pc1[..., -1] < self.max_depth, pc2[..., -1] < self.max_depth)
            pc1, pc2 = pc1[mask], pc2[mask]

            # NaN check
            mask = np.logical_not(np.isnan(np.sum(pc1, axis=-1) + np.sum(pc2, axis=-1)))
            pc1, pc2 = pc1[mask], pc2[mask]

            # compute scene_flow
            flow_3d = pc2 - pc1
            occ_mask_3d = np.zeros(len(pc1), dtype=np.bool)
        else:
            pc1 = disp2pc(disp1, baseline, f, cx, cy)
            pc2 = disp2pc(disp2, baseline, f, cx, cy)
            flow_3d = disp2pc(disp1 + disp1_change, baseline, f, cx, cy, flow_2d) - pc1

            # apply depth mask
            mask1 = (pc1[..., -1] < self.max_depth)
            mask2 = (pc2[..., -1] < self.max_depth)
            pc1, pc2, flow_3d, occ_mask_3d = pc1[mask1], pc2[mask2], flow_3d[mask1], occ_mask_2d[mask1]

            # NaN check
            mask1 = np.logical_not(np.isnan(np.sum(pc1, axis=-1) + np.sum(flow_3d, axis=-1)))
            mask2 = np.logical_not(np.isnan(np.sum(pc2, axis=-1)))
            pc1, pc2, flow_3d, occ_mask_3d = pc1[mask1], pc2[mask2], flow_3d[mask1], occ_mask_3d[mask1]

        
        # save point clouds and occ mask
        np.savez(
            os.path.join(self.output_dir, self.split, 'pc', '%07d.npz' % index1),
            pc1=pc1, 
        )

        # mask regions moving extremely fast
        flow_mask = np.logical_and(np.abs(flow_2d[..., 0]) < 500, np.abs(flow_2d[..., 1]) < 500)
        flow_2d[np.logical_not(flow_mask)] = 0.0

        # save ground-truth flow
        save_flow_path = os.path.join(self.output_dir, self.split, 'flow_2d', '%07d.png' % index1)
        save_flow_png(
            save_flow_path,
            flow_2d, flow_mask
        )
        np.save(
            os.path.join(self.output_dir, self.split, 'flow_3d', '%07d.npy' % index1),
            flow_3d
        )
        
        return save_flow_path


def main():
    for split_idx, split in enumerate(['train', 'val']):
        
        if not os.path.exists(os.path.join(args.input_dir, split)):
            continue

        logging.info('Processing "%s" split...' % split)

        os.makedirs(os.path.join(args.output_dir, split, 'pc'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, 'flow_2d'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, 'flow_3d'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, 'occ_mask_3d'), exist_ok=True)

        if not os.path.exists(os.path.join(args.output_dir, split, 'image_clean')):
            logging.info('Copying images...')
            shutil.copytree(
                src=os.path.join(args.input_dir, split, 'image_clean', 'left'),
                dst=os.path.join(args.output_dir, split, 'image_clean')
            )
        else:
            logging.info('Images already exist.')

        if not os.path.exists(os.path.join(args.output_dir, split, 'occ_mask_2d')):
            logging.info('Copying occ_mask_2d...')
            shutil.copytree(
                src=os.path.join(args.input_dir, split, 'flow_occlusions', 'left', 'into_future'),
                dst=os.path.join(args.output_dir, split, 'occ_mask_2d')
            )
        else:
            logging.info('Occ_mask_2d already exist.')

        logging.info('Generating point clouds...')
        preprocessor = Preprocessor(
            args.input_dir,
            args.output_dir,
            split,
            args.n_points[split_idx],
            args.max_depth,
            args.remove_occluded_points,
        )
        preprocessor[0]
        preprocessor = torch.utils.data.DataLoader(dataset=preprocessor, num_workers=4)

        path_list = []
        for flow_path in tqdm(preprocessor):
            abs_path = flow_path[0]
            tqdm.write(f"Saved {abs_path}")
            rel_path = abs_path.replace(f"{args.output_dir}/", '')
            path_list.append(rel_path)
        
        # NOTE: activate if you want to re-generate the split txt file
        # split_txt = f"data_split/flying_things_3D/{split}_regen.txt"
        # os.makedirs(os.path.dirname(split_txt), exist_ok=True)
        # with open(split_txt, 'w') as f:
        #     for path in path_list:
        #         f.write(path + '\n')

if __name__ == '__main__':
    init_logging()
    main()
    logging.info('All done.')
