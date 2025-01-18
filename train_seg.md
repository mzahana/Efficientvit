# Training EfficientViT for Semantic Segmentation

This guide provides instructions for:
1. Installing the necessary dependencies and repository.
2. Preparing your dataset (and converting it if needed).
3. Configuring the training hyperparameters (batch size, epochs, etc.).
4. Running the training script.
5. Monitoring training and managing checkpoints.

## 1. Installation

```bash
git clone -b dev https://github.com/mzahana/Efficientvit.git
cd Efficientvit
conda create -n efficientvit python=3.10
conda activate efficientvit
conda install -c conda-forge mpi4py openmpi
pip install -r requirements.txt
```
---

## 2. Prepare Your Dataset

### 2.1 Data Format

EfficientViT for semantic segmentation expects **per-pixel** class labels as single-channel images:
- **0** = background
- **1, 2, …** = your foreground classes

Depending on whether you want to do **K-Fold cross-validation** or a **simple train/validation split**, you have two main dataset structures:

1. **Simple Train/Validation Folders** (`SemanticSegmentationDataset`):
   ```
   dataset/
   ├── images/
   │   ├── training/
   │   │   ├── img1.jpg (or .png)
   │   │   ├── ...
   │   └── validation/
   │       ├── imgX.jpg
   │       └── ...
   └── annotations/
       ├── training/
       │   ├── img1.png (mask)
       │   ├── ...
       └── validation/
           ├── imgX.png
           └── ...
   ```

2. **Single Images + Masks Folders** (for **K-Fold** using `AutoSemanticSegmentationDataset`):
   ```
   dataset/
   ├── images/
   │   ├── image1.png
   │   ├── image2.png
   │   └── ...
   └── masks/
       ├── mask1.png
       ├── mask2.png
       └── ...
   ```

> **Need to convert from another format (e.g. YOLO polygons)**?  
> Use a script that draws polygons onto blank masks as integer IDs. See [convert_yolo2efficient.py](convert_yolo2efficient.py) or a similar tool.

---

## 3. Configuring Hyperparameters

### 3.1 Edit `train_seg_configs.py`

Inside `train_seg_configs.py`, you’ll typically find (or add) variables like:

```python
# train_seg_configs.py
import os.path as path

taskID=r'veddesta-segeltorp'
out_weights_path = path.join('checkpoints',taskID)
#TensorBoard
TensorBoard_dir=path.join('tensorboard-log',taskID)

root_dir = '/path/to/your/dataset'


max_to_save = 5 # number of model checkpoints to keep
epochs = 50
LR = 0.0001
Batch_Size = 8
Num_workers = 6
N_splits = 5    # if >1, triggers K-Fold cross-validation
Model_size = 'b0'   # efficientvit model variant: b0, b1, b2, b3, l1, l2
IS_IOU_ACC = False  # whether to compute IoU & Accuracy at validation

#早停策略 val loss没有提升的次数
Patient_Num=10  # early stopping patience
#早停策略 val loss 平滑程度
Std_Smooth=0.05

# Newly added: image size
image_size = (512, 512)

```

**Key fields**:

- `root_dir`: path to your dataset  
- `Batch_Size`: reduce if you get CUDA Out-of-Memory errors  
- `Model_size`: pick a model from `b0`, `b1`, `b2`, `b3`, `l1`, `l2`  
- `LR`: initial learning rate  
- `epochs`: total training epochs  
- `N_splits`: if >1, the script uses `KFold` for cross-validation  

### 3.2 Add Your Dataset Name in `seg_model_zoo.py` (If Needed)

If you have a **custom dataset** with a different number of classes (e.g., 3 classes), you can add or modify an entry in `seg_model_zoo.py` under `REGISTERED_SEG_MODEL`, something like:

```python
"mydataset3": {
    "b0": None,  # or a path if you have pretrained weights
    "b1": None,
    ...
},
```

And in `seg.py`, define the head with `n_classes=<my_num_classes>` under your new dataset key:

```python
# Example in efficientvit_seg_b0
elif dataset == "mydataset3":
    head = SegHead(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[128, 64, 32],
        stride_list=[32, 16, 8],
        head_stride=8,
        head_width=32,
        head_depth=1,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
        n_classes=3,
        ...
    )
```

Then, during training, use:

```python
model = create_seg_model("b0", "mydataset3", pretrained=False)
```

---

## 4. Running the Training Script

### 4.1 Standard Usage

Assuming your training script is named `train_seg.py`, you can typically do:
```bash
python train_seg.py
```
This script will:
1. Load `train_seg_configs.py`.
2. Initialize datasets (train + valid, or K-Fold).
3. Build and compile an EfficientViT segmentation model with your specified `Model_size`.
4. Train for `epochs` epochs or until early stopping triggers.
5. Save model checkpoints to `out_weights_path`.
6. Log metrics to `TensorBoard_dir`.

### 4.2 Monitor with TensorBoard

You can monitor training metrics (loss, IoU, accuracy, etc.) in real time via:
```bash
tensorboard --logdir ./tensorboard-log
```
Then open `http://localhost:6006` in your browser.

---

## 5. Model Checkpoints

- By default, the script saves a model checkpoint each epoch in `out_weights_path` (e.g. `checkpoints/my_experiment`).
- It keeps only the most recent `max_to_save` checkpoints (old ones get deleted).
- The checkpoint filenames typically follow a pattern like:
  ```
  model_epoch_1_loss_0.8500_20230101_123456.pth
  model_epoch_2_loss_0.7000_20230101_130345.pth
  ...
  ```

If you restart training, the script will automatically:
- Look for the latest checkpoint in `out_weights_path`.
- Resume training from the next epoch.

---

## 6. Changing or Customizing the Script

Some common customizations:

1. **Data Augmentation**  
   The transforms (`SegmentationTransforms`, `SegTransforms2`) are in the script. Adjust color jitter, flips, rotations as needed.

2. **Loss Function**  
   Currently `nn.CrossEntropyLoss()` is used. For class imbalance, you might switch to a weighted cross-entropy or Dice loss, etc.

3. **Scheduler**  
   The default is `ReduceLROnPlateau`. Feel free to change to a cosine scheduler or something else if needed.

4. **Early Stopping**  
   The script uses a custom `EarlyStopping` utility. Adjust the patience (`Patient_Num`) if training is being interrupted too soon or too late.

---

## 7. Inference or Evaluation

- After training, you can load a checkpoint for inference.  
- Example snippet (PyTorch-based):
  ```python
  import torch
  from efficientvit.seg_model_zoo import create_seg_model

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = create_seg_model("b0", "mydataset3", pretrained=False)
  state_dict = torch.load("checkpoints/my_experiment/model_epoch_XX_loss_YYYY.pth", map_location=device)
  model.load_state_dict(state_dict)
  model.to(device)
  model.eval()

  # then feed your test images to produce segmentation masks
  ```
- Or implement a separate script that loads the model and runs inference on single images or an entire folder.

---

## 8. Summary

**Checklist** for training EfficientViT on your dataset:

1. **Install** PyTorch & dependencies.
2. **Convert** or **prepare** your dataset (images + single-channel masks).
3. **Update** `train_seg_configs.py` with your paths and hyperparameters.
4. **Optional**: Modify `seg_model_zoo.py` or `seg.py` if you have a new dataset name or a different number of classes.
5. **Run** `python train_seg.py`.
6. **Monitor** logs in TensorBoard.
7. **Check** saved checkpoints in your `checkpoints/<taskID>` folder.

You’re all set to train and explore **EfficientViT** for semantic segmentation!