#!/usr/bin/env python3
"""
compute_mean_std.py

Usage Example:
    python compute_mean_std.py --root_dir /path/to/your/images
"""

import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Compute mean and std of a dataset of RGB images.")
    parser.add_argument('--root_dir', type=str, required=True,
                        help="Root directory containing images (can have subfolders).")
    args = parser.parse_args()

    # We'll accumulate the sums and the sums of squares for each channel separately.
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    num_pixels = 0

    # Simple transform to convert PIL -> Tensor in [0..1] range
    to_tensor = transforms.ToTensor()

    # Recursively walk the root_dir to find images
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    file_count = 0

    for root, dirs, files in os.walk(args.root_dir):
        for filename in files:
            if filename.lower().endswith(valid_exts):
                file_count += 1

    if file_count == 0:
        print(f"No images found in {args.root_dir} with extensions {valid_exts}. Exiting.")
        return

    print(f"Found {file_count} valid image(s) under {args.root_dir}. Computing mean/std...")

    current_file = 0
    for root, dirs, files in os.walk(args.root_dir):
        for filename in files:
            if filename.lower().endswith(valid_exts):
                current_file += 1
                img_path = os.path.join(root, filename)
                # Open image
                with Image.open(img_path).convert("RGB") as img:
                    img_t = to_tensor(img)  # shape: [3, H, W]
                    # Sum over height and width
                    channel_sum += img_t.sum(dim=[1, 2])
                    channel_sq_sum += (img_t ** 2).sum(dim=[1, 2])
                    num_pixels += img_t.shape[1] * img_t.shape[2]

                if current_file % 200 == 0:
                    print(f"Processed {current_file}/{file_count} images...")

    # Calculate mean and std
    # mean = total_sum / (total_number_of_pixels)
    # var = (sum_of_squares / total_number_of_pixels) - (mean^2)
    # std = sqrt(var)
    mean = channel_sum / num_pixels
    var = (channel_sq_sum / num_pixels) - (mean ** 2)
    std = torch.sqrt(var)

    print("\n=== Dataset Statistics ===")
    print(f"Mean per channel: {mean.tolist()}")  # e.g. [0.485, 0.456, 0.406]
    print(f"Std per channel:  {std.tolist()}")   # e.g. [0.229, 0.224, 0.225]

if __name__ == "__main__":
    main()
