# Forgery Detection with DualNet

A complete system for training and evaluating deep learning models that detect image forgery. The DualNet model operates as a single system that performs both classification of forgery types and localization of forged areas.

## Features

- **Dual-task Learning**: Simultaneous classification and localization
- **Advanced Architecture**: Based on EfficientNet with custom modules for forgery detection
- **Multi-class Detection**: Detects four different types of forgery:
  - Real images
  - Splicing forgeries
  - Copy-move forgeries
  - AI-generated images
- **Robust Training**:
  - Data augmentation and preprocessing
  - Early stopping
  - Checkpoint saving
- **Comprehensive Evaluation**:
  - Multiple metrics
  - Detailed visualizations
  - Performance analysis

## Requirements

### Python Dependencies

```bash
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.5
pandas>=1.3.0
scikit-learn>=0.24.2
matplotlib>=3.4.2
Pillow>=8.3.1
albumentations>=1.0.3
efficientnet-pytorch>=0.7.1
tqdm>=4.62.2
opencv-python>=4.5.3
```

### Hardware Requirements

- CUDA-capable GPU (recommended)
- 16GB RAM (Recommended)
- 20GB free disk space (for dataset and model storage)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aman3579/Dualnet.git
cd DualNet
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install packages directly:
```bash
pip install torch efficientnet_pytorch torchvision numpy pandas matplotlib scikit-learn opencv-python albumentations tqdm pillow
```

## Dataset Structure

The dataset is organized into four main categories:

```
dataset/
├── real/
│   ├── Artifact/              # Authentic images from Artifact
│   └── Cifake/               # Real images from CiFake dataset
├── splicing/
│   ├── casia/                # CASIA splicing dataset
│   │   ├── images/          # Spliced images
│   │   └── masks/           # Ground truth masks
│   └── defacto/             # DeFacto splicing dataset
│       ├── images/          # Spliced images
│       └── masks/           # Ground truth masks
├── copy_move/
│   ├── defacto/             # DeFacto copy-move dataset
│   │   ├── images/          # Copy-moved images
│   │   └── masks/           # Ground truth masks
│   └── comofod/             # CoMoFoD copy-move dataset
│       ├── images/          # Copy-moved images
│       └── masks/           # Ground truth masks
└── aigenerated/
    ├── Artifact/            # AI-generated images from Artifact dataset
    └── Cifake/              # AI-generated images from CiFake dataset
```

## Dataset Splits
Run python preprocessing_new.py to preprocess the dataset.
It will save train, val, test splits into csv files. 

The `splits/` directory contains automatically generated CSV files that organize the data for training:

### Classification Splits
```
splits/
├── classification_train.csv  # 70% of data
├── classification_val.csv    # 15% of data
└── classification_test.csv   # 15% of data
```
- Includes all four categories
- Balanced distribution across classes

### Localization Splits
```
splits/
├── localization_train.csv    # 70% of data with masks
├── localization_val.csv      # 15% of data with masks
└── localization_test.csv     # 15% of data with masks
```

## Usage

### Training

```bash
python train_new.py \
    --dataset /path/to/dataset \
    --epochs 400 \
    --batch_size 16 \
    --use_albumentations \
    --output_dir ./runs
```

### Evaluation

```bash
python evaluate_new.py \
    --model /path/to/model.pth \
    --dataset /path/to/dataset \
    --output evaluation_results
```

### Key Training Parameters

- `--epochs`: Number of training epochs (default: 400)
- `--batch_size`: Batch size (default: 16)
- `--use_albumentations`: Enable advanced augmentations
- `--memory_efficient`: Enable memory optimizations
- `--save_lite`: Save lightweight checkpoints
- `--compress_ckpt`: Compress saved checkpoints
- `--best_only`: Save only best checkpoints

## Model Architecture

### Classification Branch
- EfficientNet-B4 backbone
- Frequency Analysis and Harmonic Attention (FAHA)
- Multi-scale Gradient Consistency Module (MGCM)
- Dual attention fusion with transformer blocks

### Localization Branch
- Enhanced Laplacian module
- Spatial attention gates
- Multi-scale feature fusion
- U-Net style decoder

## Performance Metrics

### Classification
- Accuracy
- F1 Score
- Precision/Recall
- AUROC
- Confusion Matrix

### Localization
- IoU (Intersection over Union)
- Dice Coefficient
- F1 Score
- Precision/Recall
- AUROC
- Average Precision

## Visualization Tools

The evaluation script generates:
- Classification visualizations with confidence scores
- Localization heatmaps
- Confusion matrices
- Sample predictions with ground truth