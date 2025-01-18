""" 
python train_seg.py --dataset_mode auto

or

python train_seg.py --dataset_mode standard
"""
import argparse
import sys
import os
import re
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.metrics import jaccard_score, accuracy_score
from PIL import Image

# Transformers for the SegformerImageProcessor
from transformers import SegformerImageProcessor

# Import your configs
from configs.seg.train_seg_configs import (
    root_dir,
    out_weights_path,
    TensorBoard_dir,
    max_to_save,
    epochs,
    LR,
    Batch_Size,
    Num_workers,
    N_splits,
    Model_size,
    IS_IOU_ACC,
    image_size
)

# Import EarlyStopping from efficientvit
from efficientvit.tools.early_stopping import EarlyStopping

# Import the create_seg_model function
from efficientvit.seg_model_zoo import create_seg_model

# Dataset classes (adjust import paths to where you defined them)
# Make sure you have these classes in your codebase:
#   - SemanticSegmentationDataset (for standard train/val)
#   - AutoSemanticSegmentationDataset (for K-Fold cross-validation)
class SemanticSegmentationDataset(torch.utils.data.Dataset):
    """Image (semantic) segmentation dataset: expects
       root_dir/images/training, root_dir/images/validation
       root_dir/annotations/training, root_dir/annotations/validation, etc.
    """
    def __init__(self, root_dir, image_processor, transform=None, train=True):
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.transform = transform
        self.train = train
        
        sub_path = "training" if self.train else "validation"
        self.img_dir = os.path.join(self.root_dir, "images", sub_path)
        self.ann_dir = os.path.join(self.root_dir, "annotations", sub_path)

        image_file_names = []
        for _, _, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

        annotation_file_names = []
        for _, _, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), \
            "Mismatch between number of images and annotation masks."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        ann_path = os.path.join(self.ann_dir, self.annotations[idx])

        image = Image.open(img_path).convert("RGB")
        seg_map = Image.open(ann_path).convert("L")

        if self.transform:
            image, seg_map = self.transform(image, seg_map)

        encoded_inputs = self.image_processor(
            image, seg_map, return_tensors="pt"
        )
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension
        return encoded_inputs

class AutoSemanticSegmentationDataset(torch.utils.data.Dataset):
    """Image (semantic) segmentation dataset for K-Fold usage: expects
       root_dir/images/*.png, root_dir/masks/*.png (paired).
    """
    def __init__(self, root_dir, image_processor, transform=None, train=True):
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.transform = transform

        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")

        image_file_names = []
        for _, _, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

        mask_file_names = []
        for _, _, files in os.walk(self.mask_dir):
            mask_file_names.extend(files)
        self.masks = sorted(mask_file_names)

        assert len(self.images) == len(self.masks), \
            "Mismatch between number of images and masks."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        seg_map = Image.open(mask_path).convert("L")

        if self.transform:
            image, seg_map = self.transform(image, seg_map)

        encoded_inputs = self.image_processor(
            image, seg_map, return_tensors="pt"
        )
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension
        return encoded_inputs

# If you have data augmentation transforms, define them here
# e.g. transform = SegmentationTransforms() or SegTransforms2() from your code
transform = None

# Global list for storing losses
loss_list = []

def compute_metrics(preds, labels, num_classes=12):
    """Compute IoU and Accuracy using sklearn.metrics."""
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    iou = jaccard_score(labels_flat, preds_flat, average='macro', labels=range(num_classes))
    accuracy = accuracy_score(labels_flat, preds_flat)
    return iou, accuracy

def get_epoch(filename):
    match = re.search(r"model_epoch_(\d+)_", filename)
    if match:
        return int(match.group(1))
    else:
        return None

def delete_min_epoch_file(directory):
    files = os.listdir(directory)
    epochs = [get_epoch(f) for f in files if f.endswith(".pth")]
    if epochs:
        min_epoch = min(epochs)
        for filename in files:
            if get_epoch(filename) == min_epoch:
                os.remove(os.path.join(directory, filename))
                print(f"[INFO] Deleted old checkpoint: {filename}")
                break

def load_checkpoint(model, checkpoint_dir) -> int:
    """Load the latest checkpoint from checkpoint_dir. Returns the epoch number."""
    if not os.path.isdir(checkpoint_dir):
        print("[INFO] Checkpoint directory not found; starting from scratch.")
        return 0
    checkpoint_files = [os.path.join(checkpoint_dir, f)
                        for f in os.listdir(checkpoint_dir)
                        if f.endswith('.pth')]
    if checkpoint_files:
        # Sort by epoch
        checkpoint_files = sorted(checkpoint_files, key=lambda f: float(f.split('_')[2]))
        latest_checkpoint_path = checkpoint_files[-1]
        print(f"[INFO] Loading checkpoint from {latest_checkpoint_path}")
        epoch = int(latest_checkpoint_path.split('_')[2])
        model.load_state_dict(torch.load(latest_checkpoint_path))
        return epoch
    else:
        print("[INFO] No checkpoint found; starting from scratch.")
        return 0

def save_checkpoint(model, checkpoint_dir, epoch, avg_val_loss, max_to_save):
    """Save current checkpoint, keep only last max_to_save."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = f'model_epoch_{epoch}_loss_{avg_val_loss:.4f}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
    model_path = os.path.join(checkpoint_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Saved model checkpoint: {model_path}")

    loss_list.append(avg_val_loss)
    if len(loss_list) > max_to_save:
        delete_min_epoch_file(checkpoint_dir)

def main():
    parser = argparse.ArgumentParser(description="Train EfficientViT with optional K-Fold or standard dataset.")
    parser.add_argument("--dataset_mode", type=str, default="standard", 
                        choices=["auto", "standard"],
                        help="Which dataset mode to use. 'auto' => K-Fold with AutoSemanticSegmentationDataset, 'standard' => separate train/val (SemanticSegmentationDataset).")
    args = parser.parse_args()

    # 1. Create the image processor using the image_size from train_seg_configs.py
    image_processor = SegformerImageProcessor(
        reduce_labels=False,
        size=image_size  # from train_seg_configs.py
    )

    # 2. Check dataset_mode and build dataset(s)
    if args.dataset_mode == "auto":
        print("[INFO] Using AutoSemanticSegmentationDataset (K-Fold).")
        full_dataset = AutoSemanticSegmentationDataset(
            root_dir=root_dir,
            image_processor=image_processor,
            transform=transform,
            train=True
        )

        # K-Fold setup
        kf = KFold(n_splits=N_splits, shuffle=True, random_state=42)
        folds = list(kf.split(full_dataset))

    else:
        print("[INFO] Using SemanticSegmentationDataset (standard train/val).")
        train_dataset = SemanticSegmentationDataset(
            root_dir=root_dir,
            image_processor=image_processor,
            transform=transform,
            train=True
        )
        valid_dataset = SemanticSegmentationDataset(
            root_dir=root_dir,
            image_processor=image_processor,
            transform=transform,
            train=False
        )
        # We'll create a dummy single "fold" pairing all train indices with valid indices
        folds = [(
            list(range(len(train_dataset))),
            list(range(len(valid_dataset)))
        )]

    # 3. Create the output/checkpoint directories if needed
    os.makedirs(out_weights_path, exist_ok=True)
    os.makedirs(TensorBoard_dir, exist_ok=True)

    # 4. Loop over folds
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n[INFO] Starting Fold {fold_idx+1}/{len(folds)}")

        # 4a. Build DataLoaders
        if args.dataset_mode == "auto":
            # We have one dataset, so we use SubsetRandomSampler
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_dataloader = DataLoader(full_dataset,
                                          sampler=train_sampler,
                                          batch_size=Batch_Size,
                                          num_workers=Num_workers,
                                          pin_memory=True)
            valid_dataloader = DataLoader(full_dataset,
                                          sampler=val_sampler,
                                          batch_size=Batch_Size,
                                          num_workers=Num_workers,
                                          pin_memory=True)
        else:
            # standard dataset
            train_dataloader = DataLoader(train_dataset,
                                          batch_size=Batch_Size,
                                          shuffle=True,
                                          num_workers=Num_workers,
                                          pin_memory=True)
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=Batch_Size,
                                          shuffle=False,
                                          num_workers=Num_workers,
                                          pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

        # 4b. Create model
        model = create_seg_model(Model_size, "bundle2", pretrained=False, weight_url='/home/mzahana/src/Efficientvit/checkpoints/seg/ade20k/l2.pt')
        model.to(device)

        # 4c. Load checkpoint if any
        start_epoch = load_checkpoint(model, out_weights_path) + 1

        # 4d. Define criterion, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.1, verbose=True
        )

        # 4e. Early stopping, TensorBoard
        early_stopping = EarlyStopping()
        writer = SummaryWriter(log_dir=TensorBoard_dir)

        # 4f. Training loop
        for epoch in range(start_epoch, epochs):
            model.train()
            print(f"\nEpoch {epoch} (Fold {fold_idx+1})")
            epoch_loss = 0.0

            for batch in tqdm(train_dataloader, desc="Training"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(pixel_values)
                upsampled_outputs = nn.functional.interpolate(
                    outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = criterion(upsampled_outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_dataloader)
            print(f"[TRAIN] Epoch {epoch} - Loss: {avg_train_loss:.4f}")
            writer.add_scalar('Loss/train', avg_train_loss, epoch)

            # Validation
            model.eval()
            val_loss = 0.0
            val_ious, val_accs = [], []
            with torch.no_grad():
                for batch in tqdm(valid_dataloader, desc="Validating"):
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(pixel_values)
                    upsampled_outputs = nn.functional.interpolate(
                        outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False
                    )
                    loss = criterion(upsampled_outputs, labels)
                    val_loss += loss.item()

                    pred = upsampled_outputs.argmax(dim=1)
                    if IS_IOU_ACC:
                        iou, acc = compute_metrics(pred.cpu().numpy(), labels.cpu().numpy())
                        val_ious.append(iou)
                        val_accs.append(acc)

            avg_val_loss = val_loss / len(valid_dataloader)
            print(f"[VAL] Epoch {epoch} - Loss: {avg_val_loss:.4f}")
            writer.add_scalar('Loss/val', avg_val_loss, epoch)

            if IS_IOU_ACC and val_ious:
                mean_iou = sum(val_ious) / len(val_ious)
                mean_acc = sum(val_accs) / len(val_accs)
                print(f"[VAL] IoU: {mean_iou:.4f}, Accuracy: {mean_acc:.4f}")
                writer.add_scalar('IoU/val', mean_iou, epoch)
                writer.add_scalar('Accuracy/val', mean_acc, epoch)

            # Scheduler
            scheduler.step(avg_val_loss)

            # Early Stopping
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("[INFO] Early stopping triggered.")
                break

            # Save checkpoint
            save_checkpoint(model, out_weights_path, epoch, avg_val_loss, max_to_save)

        # End of epoch loop
        print(f"[INFO] Fold {fold_idx+1} training complete.")
        writer.close()
        # If you want to proceed with next fold from scratch or from a certain checkpoint,
        # you can adjust logic here.

    # End of folds
    print("[INFO] Training process finished.")


if __name__ == "__main__":
    # For Linux, you might also want to use torch.compile if desired
    # Just ensure PyTorch version >= 2.0
    # e.g. model = torch.compile(model) in the code above
    main()
