import argparse
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from . import model
from .utils import general, landmark_tools
import pickle
import pickle5
import cv2
import shutil
from glob import glob
from tqdm import tqdm

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='data/Classification/JPEGImages/', help="path to image directory")
parser.add_argument('--landmark_path', type=str, default='data/Results/', help="path to landmark directory")
parser.add_argument('--Gckpt', type=str, default=None, help="Generator Checkpoint")
parser.add_argument('--save_dir', type=str, default=None, help="Folder to save generated images")
parser.add_argument('--num_images', type=int, default=1, help="Number of images to process")
parser.add_argument('--num_patch', type=int, default=1, help="Number of patches to modify for an image")
parser.add_argument('--image_size', default=64, type=int, help="image patch size")
parser.add_argument('--hidden_size', default=64, type=int, help="network hidden unit size")
parser.add_argument('--num_input_channel', default=3, type=int, help="for RGB, it is 3")
parser.add_argument('--patch_size_ratio', default=15, type=int)
parser.add_argument('--patch_filter_magic', default=1, type=float)
parser.add_argument('--gpu', default='', type=str, help="the id of gpu(s) to use")
args = parser.parse_args()

# Process Image Function
def process_image(image_path, landmark_path, save_dir, num_patch=1):
    # Load image and landmark
    img = plt.imread(image_path)
    with open(landmark_path, 'rb') as fp:
        try:
            landmark_list = pickle.load(fp)
        except ValueError:
            landmark_list = pickle5.load(fp)
    tri_params = landmark_tools.obtain_preprocess_triangles(img, landmark_list)

    # Create a copy of the original image to apply all patches
    synimage_all_patches = img.copy()

    for I in range(num_patch):
        # Generate patch
        ptr, ori_patch = landmark_tools.generate_filtered_patch(img, tri_params, 
                                                               size_ratio=args.patch_size_ratio, 
                                                               filter_magic=args.patch_filter_magic)
        origin_size = ori_patch.shape[:2]

        # Resize and prepare patch for the network
        resized_patch = cv2.resize(ori_patch, (args.image_size, args.image_size))
        patch = torch.FloatTensor(resized_patch) / 255.
        patch = patch.permute([2, 0, 1])
        patch = torch.unsqueeze(patch, 0)  # 1, 3, H, W
        patch.to(device)

        # Generate synthetic patch
        synpatch, modify, mask = netG(patch)

        def process_image(tensor):
            tensor = general.to_numpy(tensor)
            tensor = np.transpose(tensor, [0, 2, 3, 1])
            tensor = tensor[0]
            return tensor
        
        synpatch = process_image(synpatch)

        # Resize the synthesized patch back to its original size
        synpatch_fullsize = cv2.resize(synpatch, origin_size)
        p = (synpatch_fullsize * 255).astype(img.dtype)

        # Apply synthesized patch to the copy of the original image
        size = origin_size[0] // 2
        synimage_all_patches[ptr[1]-size:ptr[1]+size, ptr[0]-size:ptr[0]+size] = p

    # Save the final image with all patches applied
    fname = os.path.basename(image_path)
    fname = '.'.join(fname.split('.')[:-1])  # remove file extension
    plt.imsave(os.path.join(save_dir, f"{fname}_{num_patch}_patches.jpg"), synimage_all_patches)

    # Copy the original image to the save directory
    shutil.copy(image_path, os.path.join(save_dir, f"{fname}.jpg"))

# Main Execution
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if len(args.gpu) > 0 else "cpu")
    netG = model.Generator(args.hidden_size, num_input_channel=args.num_input_channel)
    ckpt = torch.load(args.Gckpt, map_location=device)
    netG.load_state_dict(ckpt)
    netG.eval()
    netG.to(device)

    os.makedirs(args.save_dir, exist_ok=True)

    # Select and process images based on specific landmark files
    landmark_files = glob(os.path.join(args.landmark_path, "levle0_*.pkl"))
    random.shuffle(landmark_files)
    processed_images = 0

    for landmark_file in landmark_files:
        if processed_images >= args.num_images:
            break

        image_file = os.path.join(args.image_path, os.path.basename(landmark_file).replace('.pkl', '.jpg'))
        if os.path.exists(image_file):
            process_image(image_file, landmark_file, args.save_dir, args.num_patch)
            processed_images += 1
