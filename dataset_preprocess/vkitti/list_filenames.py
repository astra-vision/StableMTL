import os
import tqdm

def main():
    split = "val"
    filename_ls_path = f"data_split/vkitti/vkitti_{split}.txt"
    dataset_dir = "/lustre/fsn1/projects/rech/kvd/uyl37fq/stablepoop_processed/vkitti_v2"
    with open(filename_ls_path, "r") as f:
        filenames = [s.strip().split() for s in f.readlines()]

    # File paths for output text files
    vkitti_semantic_txt = f"data_split/vkitti/vkitti_{split}_semantic.txt"
    vkitti_normal_txt = f"data_split/vkitti/vkitti_{split}_normal.txt"
    vkitti_depth_txt = f"data_split/vkitti/vkitti_{split}_depth.txt"
    vkitti_optical_flow_txt = f"data_split/vkitti/vkitti_{split}_optical_flow.txt"

    # Open output files once before the loop
    with open(vkitti_semantic_txt, 'w') as semantic_file, \
         open(vkitti_normal_txt, 'w') as normal_file, \
         open(vkitti_depth_txt, 'w') as depth_file, \
         open(vkitti_optical_flow_txt, 'w') as optical_flow_file:

        for filename in tqdm.tqdm(filenames):
            rgb_rel_path = filename[0]
            rgb_rel_path = os.path.join("original", rgb_rel_path)

            depth_rel_path = os.path.join("original", filename[1])
            semantic_rel_path = rgb_rel_path.replace("/rgb/", "/classSegmentation/").replace('rgb_', 'classgt_').replace('jpg', 'png')
            normal_rel_path = depth_rel_path.replace('original', 'normal_estimated').replace('png', 'npy').replace('depth', 'normal')
            optical_flow_rel_path = rgb_rel_path.replace('rgb_', 'flow_').replace('rgb', 'forwardFlow').replace('jpg', 'png')

            # Check if files exist and print with the message
            if not os.path.exists(os.path.join(dataset_dir, rgb_rel_path)):
                raise ValueError(f"Not found: {os.path.join(dataset_dir, rgb_rel_path)}")
        
            if not os.path.exists(os.path.join(dataset_dir, depth_rel_path)):
                print(f"Not found: {os.path.join(dataset_dir, depth_rel_path)}")
            else:
                depth_file.write(depth_rel_path + '\n')
                
            if not os.path.exists(os.path.join(dataset_dir, semantic_rel_path)):
                print(f"Not found: {os.path.join(dataset_dir, semantic_rel_path)}")
            else:
                semantic_file.write(semantic_rel_path + '\n')
                
            if not os.path.exists(os.path.join(dataset_dir, normal_rel_path)):
                print(f"Not found: {os.path.join(dataset_dir, normal_rel_path)}")
            else:
                normal_file.write(normal_rel_path + '\n')
                
            if not os.path.exists(os.path.join(dataset_dir, optical_flow_rel_path)):
                print(f"Not found: {os.path.join(dataset_dir, optical_flow_rel_path)}")
            else:
                optical_flow_file.write(optical_flow_rel_path + '\n')
                


if __name__ == "__main__":
    main()
