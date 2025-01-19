import os
import random
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def generate_class_colors(num_classes):
    """Generate a unique color for each class."""
    np.random.seed(42)  # For consistent colors across runs
    return [tuple(np.random.randint(0, 256, 3)) for _ in range(num_classes)]

def overlay_mask_on_image(image, mask, class_colors, alpha=0.5, ignore_class=None):
    """Overlay the segmentation mask on the original image."""
    image = np.array(image)
    mask = np.array(mask)

    overlay = image.copy()
    for class_idx, color in enumerate(class_colors):
        if ignore_class is not None and class_idx == ignore_class:
            continue
        overlay[mask == class_idx] = (1 - alpha) * image[mask == class_idx] + alpha * np.array(color)

    return Image.fromarray(overlay.astype(np.uint8))

def plot_random_images(dataset_dir, split_type, num_images, num_classes, ignore_class=None):
    # Define paths for images and annotations
    image_path = os.path.join(dataset_dir, 'images', split_type)
    annotation_path = os.path.join(dataset_dir, 'annotations', split_type)
    
    # Check if paths exist
    if not os.path.exists(image_path) or not os.path.exists(annotation_path):
        raise ValueError(f"Invalid split type or dataset directory. Check paths: {image_path} and {annotation_path}")

    # Get list of files
    image_files = sorted([f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    annotation_files = sorted([f for f in os.listdir(annotation_path) if f.endswith('.png')])

    if len(image_files) == 0 or len(annotation_files) == 0:
        raise ValueError("No valid image or annotation files found in the specified directories.")

    # Select random indices
    num_images = min(num_images, len(image_files))
    selected_indices = random.sample(range(len(image_files)), num_images)

    # Generate class colors
    class_colors = generate_class_colors(num_classes)

    # Create save path
    save_path = os.path.join(dataset_dir, 'plots', split_type)
    os.makedirs(save_path, exist_ok=True)

    for idx in selected_indices:
        # Load image and annotation
        image_file = image_files[idx]
        annotation_file = annotation_files[idx]
        
        image = Image.open(os.path.join(image_path, image_file)).convert("RGB")
        annotation = Image.open(os.path.join(annotation_path, annotation_file))

        # Get maximum value in the mask
        max_value = np.array(annotation).max()
        print(f"Maximum value in {annotation_file}: {max_value}")

        # Overlay mask on image
        overlay = overlay_mask_on_image(image, annotation, class_colors, ignore_class=ignore_class)

        # Plot and save the overlay
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(overlay)
        ax.set_title(f"Image with Overlay: {image_file}")
        ax.axis('off')

        # Save the plot
        plot_filename = os.path.join(save_path, f"{os.path.splitext(image_file)[0]}_overlay.png")
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {num_images} overlay plots in {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot random semantic segmentation images with overlaid masks.")
    parser.add_argument('dataset_dir', type=str, help="Path to the dataset directory.")
    parser.add_argument('split_type', type=str, choices=['training', 'validation', 'test'], help="Dataset split type.")
    parser.add_argument('num_images', type=int, help="Number of random images to plot.")
    parser.add_argument('num_classes', type=int, help="Number of classes in the dataset.")
    parser.add_argument('--ignore_class', type=int, default=None, help="Class value to ignore from plotting (e.g., 0).")

    args = parser.parse_args()
    
    plot_random_images(args.dataset_dir, args.split_type, args.num_images, args.num_classes, args.ignore_class)
