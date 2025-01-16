#!/usr/bin/env python3

"""
Converts semantic segmentation dataset from YOLO format to EfficientViT format
Deps:
    pip install pyyaml tqdm pillow

python convert_yolo2efficient.py \
    --source /path/to/yolo_dataset \
    --output /path/to/output_folder \
    --norm
    
If your coordinates are absolute, do:
    python convert_yolo2efficient.py --source ... --no-norm
"""

import argparse
import os
import sys
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="""
    Convert a YOLO polygon dataset (with train/valid/test subfolders and data.yaml) 
    into an EfficientViT-compatible semantic segmentation dataset.
    """)
    
    parser.add_argument('--source', '-s', type=str, required=True,
                        help='Path to the YOLO dataset folder containing data.yaml and subfolders train, valid, test.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for the converted dataset. If not provided, a new folder is created.')
    parser.add_argument('--norm', action='store_true', default=True,
                        help='Whether the polygon coordinates are normalized [default: True]. '
                             'Use --norm to indicate they are normalized, or --no-norm to indicate absolute coordinates.')
    parser.add_argument('--no-norm', action='store_false', dest='norm',
                        help='Use this if the polygon coordinates are absolute pixel coordinates instead of normalized.')
    
    return parser.parse_args()


def load_data_yaml(source_path):
    """
    Load data.yaml from source_path.
    Returns a dict with keys like ['train', 'val', 'test', 'nc', 'names'] if present.
    """
    data_yaml_path = os.path.join(source_path, 'data.yaml')
    if not os.path.isfile(data_yaml_path):
        print(f"[ERROR] data.yaml not found in {source_path}")
        sys.exit(1)
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    return data_config


def check_and_get_subfolders(source_path):
    """
    Expects train/, valid/, test/ subfolders inside source_path.
    Each of these should have images/ and labels/ subfolders.
    Return a dict with:
      {
        'train': {'images': <path>, 'labels': <path>},
        'valid': {'images': <path>, 'labels': <path>},
        'test':  {'images': <path>, 'labels': <path>}
      }
    If any are missing, prints an error and exits.
    """
    subsets = ['train', 'valid', 'test']
    sub_paths = {}
    for subset in subsets:
        images_path = os.path.join(source_path, subset, 'images')
        labels_path = os.path.join(source_path, subset, 'labels')
        # It's possible a user doesn't have a test set; let's handle that gracefully
        if not os.path.exists(os.path.join(source_path, subset)):
            # subset folder doesn't exist (e.g. test might be missing). Skip it.
            continue
        
        # Now check images and labels subfolders
        if not os.path.isdir(images_path) or not os.path.isdir(labels_path):
            print(f"[WARNING] Subfolder {subset} is missing images/ or labels/. Skipping.")
            continue
        
        sub_paths[subset] = {
            'images': images_path,
            'labels': labels_path
        }
    return sub_paths


def verify_label_files(images_dir, labels_dir):
    """
    Ensure that for every image file in images_dir,
    there's a corresponding .txt file in labels_dir (same basename).
    Return the list of valid image filenames that have matching .txt.
    """
    valid_images = []
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(image_exts):
            continue
        
        base, _ = os.path.splitext(fname)
        label_file = os.path.join(labels_dir, base + '.txt')
        if not os.path.isfile(label_file):
            # Possibly raise an error or skip
            print(f"[WARNING] Missing label file for image {fname} -> skipping.")
            continue
        valid_images.append(fname)
    
    return valid_images


def yolo_polygon_to_mask(image_path, label_path, out_mask_path, normalized=True, class_offset=1):
    """
    Convert a single YOLO polygon label to a semantic mask.
    :param image_path: path to the original image
    :param label_path: path to the .txt file with polygon info
    :param out_mask_path: path to save the resulting mask (PNG)
    :param normalized: if True, x/y are in [0, 1], else absolute pixel coords
    :param class_offset: if you have background=0, then class 0 => mask 1, class 1 => mask 2, etc.
    """
    # Open image to get width, height
    with Image.open(image_path) as img:
        w, h = img.size

    # Create a blank mask
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # Read each line in the label file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Format: class_id x0,y0,x1,y1,x2,y2,...
        parts = line.split()
        class_id = int(parts[0])
        # The rest is a single string with comma-separated coords or multiple floats
        # YOLOv8 polygons can vary in format. Let's assume each subsequent item is xN,yN in pairs.
        # So let's parse them carefully if they are space-delimited or comma-delimited
        # Example: "0 0.1,0.2,0.15,0.25,0.2,0.3"
        coords_str = parts[1:]  # might be a single or multiple
        coords_str = " ".join(coords_str)  # combine if splitted
        # Now split by comma
        coords_values = coords_str.split(',')
        
        # Another approach: if YOLO stored them as space separated?
        # It's best to confirm the exact format. We'll assume they're comma-separated.
        polygon_points = []
        # coords_values should be [x0, y0, x1, y1, ...]
        if len(coords_values) < 4:
            # Not a valid polygon
            continue

        # Convert to numeric
        coords_values = [float(c) for c in coords_values]

        # Build the polygon list
        # If normalized, multiply by w, h
        for i in range(0, len(coords_values), 2):
            x = coords_values[i]
            y = coords_values[i+1]
            if normalized:
                x *= w
                y *= h
            polygon_points.append((x, y))
        
        fill_val = class_id + class_offset
        # Draw polygon
        draw.polygon(polygon_points, outline=fill_val, fill=fill_val)

    mask.save(out_mask_path, "PNG")


def convert_split(
    images_dir, 
    labels_dir, 
    out_img_dir, 
    out_mask_dir, 
    normalized, 
    class_offset=1
):
    """
    Convert a single split (train/valid/test) from YOLO polygons to images + masks
    in the EfficientViT-compatible directory structure.
    Returns the number of images successfully converted.
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    valid_imgs = verify_label_files(images_dir, labels_dir)
    # Show progress bar
    for fname in tqdm(valid_imgs, desc=f"Converting {os.path.basename(os.path.dirname(images_dir))}", ncols=80):
        base, ext = os.path.splitext(fname)
        src_img_path = os.path.join(images_dir, fname)
        src_lbl_path = os.path.join(labels_dir, base + '.txt')

        # Copy or move the original image to out_img_dir
        out_img_path = os.path.join(out_img_dir, fname)
        shutil.copy2(src_img_path, out_img_path)

        # Convert polygon to mask
        out_mask_path = os.path.join(out_mask_dir, base + '.png')
        yolo_polygon_to_mask(src_img_path, src_lbl_path, out_mask_path, normalized=normalized, class_offset=class_offset)

    return len(valid_imgs)


def main():
    args = parse_args()
    source_dir = args.source
    output_dir = args.output
    normalized = args.norm

    # 1. Check source folder & data.yaml
    if not os.path.isdir(source_dir):
        print(f"[ERROR] Source directory '{source_dir}' does not exist or is not a directory.")
        sys.exit(1)

    data_config = load_data_yaml(source_dir)
    # data.yaml is loaded

    # 2. Get number of classes from data.yaml
    if 'nc' not in data_config:
        print("[ERROR] 'nc' not found in data.yaml. Cannot determine number of classes.")
        sys.exit(1)
    num_classes = int(data_config['nc'])

    # If you have e.g. background + 2 classes => background=0, class1=1, class2=2
    # So class_offset=1 for your polygons
    class_offset = 1

    # 3. Setup output directory
    if output_dir is None:
        # default: create a new folder next to source
        parent_dir = os.path.dirname(os.path.abspath(source_dir))
        base_source = os.path.basename(os.path.normpath(source_dir))
        converted_name = f"{base_source}_converted"
        output_dir = os.path.join(parent_dir, converted_name)

    os.makedirs(output_dir, exist_ok=True)

    # 4. Figure out subfolders in YOLO dataset
    subsets_dict = check_and_get_subfolders(source_dir)
    # subsets_dict might have keys 'train', 'valid', 'test' if they exist

    # 5. We'll create 'images/training', 'annotations/training', etc.
    #    or 'images/validation', 'annotations/validation', etc.
    #    or skip if subset not present
    subset_to_new_name = {
        'train': 'training',
        'valid': 'validation',
        'test':  'test'
    }

    total_converted = 0
    subset_counts = {}

    for subset, paths in subsets_dict.items():
        images_dir = paths['images']
        labels_dir = paths['labels']

        new_subset_name = subset_to_new_name.get(subset, None)
        if not new_subset_name:
            continue

        out_img_dir = os.path.join(output_dir, 'images', new_subset_name)
        out_mask_dir = os.path.join(output_dir, 'annotations', new_subset_name)

        count = convert_split(
            images_dir=images_dir,
            labels_dir=labels_dir,
            out_img_dir=out_img_dir,
            out_mask_dir=out_mask_dir,
            normalized=normalized,
            class_offset=class_offset
        )
        subset_counts[subset] = count
        total_converted += count

    # 6. Print summary
    print("\nConversion Summary:")
    for subset, count in subset_counts.items():
        print(f"  {subset}: {count} images converted.")
    print(f"  Total: {total_converted} images converted across all subfolders.\n")
    print(f"Output dataset located at: {output_dir}")


if __name__ == "__main__":
    main()
