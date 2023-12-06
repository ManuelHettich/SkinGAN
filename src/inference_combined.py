import argparse
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from . import model
from .utils import general, landmark_tools
from .utils.dataset import Dataset, load_imagelist_landmark
import pickle
import pickle5
import cv2
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--image_name', type=str, default=None, help="image name")
parser.add_argument('--image_path', type=str, default=None, help="path to image")
parser.add_argument('--landmark_path', type=str, default=None, help="path to landmark")
parser.add_argument('--Gckpt', type=str, default=None, help="Generator Checkpoint")
parser.add_argument('--save_dir', type=str, default=None, help="Folder to save generated images")
parser.add_argument('--num_patch', type=int, default=1, help="Number of patches to modify for a image")

parser.add_argument('--image_size', default=64, type=int, help="image patch size")
parser.add_argument('--hidden_size', default=64, type=int, help="network hidden unit size")
parser.add_argument('--num_input_channel', default=3, type=int, help="for RGB, it is 3")

parser.add_argument('--patch_size_ratio', default=15, type=int)
parser.add_argument('--patch_filter_magic', default=1, type=float)

parser.add_argument('--gpu', default='', type=str, help="the id of gpu(s) to use")

args = parser.parse_args()



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

        # Optional: Save comparison images for individual patches
        # ...


    # Save the final image with all patches applied
    fname = os.path.basename(image_path)
    fname = '.'.join(fname.split('.')[:-1])  # remove file extension
    plt.imsave(os.path.join(save_dir, f"{fname}_{num_patch}_patches.jpg"), synimage_all_patches)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if len(args.gpu) > 0 else "cpu")
    netG = model.Generator(args.hidden_size, num_input_channel=args.num_input_channel)
    ckpt = torch.load(args.Gckpt, map_location= device)
    netG.load_state_dict(ckpt)
    netG.eval()
    netG.to(device)

    os.makedirs(args.save_dir, exist_ok=True)
    
    image_dir_path = "data/Classification/JPEGImages/"
    image_extension = ".jpg"
    landmark_dir_path = "data/Results/"
    landmark_extension = ".pkl"
    
    process_image(image_dir_path + args.image_name + image_extension,
                  landmark_dir_path + args.image_name + landmark_extension,
                  args.save_dir,
                  args.num_patch)


