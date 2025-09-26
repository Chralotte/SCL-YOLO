# SCL-YOLO: A Lightweight Model Based on Improved YOLOv11n and its Application in Blood Cell Object Detection

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## Overview

SCL-YOLO is a lightweight yet high-precision blood cell detection model based on the YOLOv11n architecture. This repository contains the official implementation of our paper "SCL-YOLO: A Lightweight Model Based on Improved YOLOv11n and its Application in Blood Cell Object Detection".

### Key Features

- **🔬 Lightweight Design**: 32% fewer parameters and 49% lower FLOPs compared to YOLOv11n
- **📈 High Accuracy**: Achieves 97.6% mAP50 and 85.2% mAP50-95 on TXL_PBC dataset
- **🚀 Real-time Performance**: Optimized for resource-constrained environments
- **🩸 Medical Application**: Specialized for automated blood cell detection

### Architecture Innovations

1. **StarNet_s050 Backbone**: Lightweight feature extraction with depthwise separable convolutions
2. **CAPFFN (Context-Aware Pyramid Feature Fusion Network)**: Enhanced multi-scale feature fusion
3. **LEDH (Lightweight Efficient Detection Head)**: Group convolution-based detection head

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 12.1+ (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/SCL-YOLO.git
cd SCL-YOLO

# Create virtual environment
conda create -n scl-yolo python=3.10
conda activate scl-yolo

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### TXL_PBC Dataset

1. Download the TXL_PBC dataset from [TXL-PBC Dataset](https://github.com/lugan113/TXL-PBC_Dataset)
2. Organize the dataset structure:

```
datasets/
├── TXL_PBC/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── CBC/
    ├── images/
    └── labels/
```

### CBC Dataset

1. Download the CBC dataset from [CBC Dataset](https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset)
2. Follow the same structure as above

## Training

### Basic Training

```bash
python train.py --config configs/scl_yolo.yaml --data datasets/TXL_PBC.yaml
```

### Training Parameters

```bash
python train.py \
    --model scl_yolo \
    --data TXL_PBC \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640 \
    --device 0 \
    --optimizer AdamW \
    --lr 0.01
```

## Evaluation

### Test on TXL_PBC Dataset

```bash
python test.py --weights runs/train/exp/weights/best.pt --data datasets/TXL_PBC.yaml
```

### Test on CBC Dataset

```bash
python test.py --weights runs/train/exp/weights/best.pt --data datasets/CBC.yaml
```

## Inference

### Single Image Inference

```bash
python detect.py --weights weights/scl_yolo.pt --source path/to/image.jpg --save-txt
```

### Batch Inference

```bash
python detect.py --weights weights/scl_yolo.pt --source path/to/images/ --save-txt
```

## Model Zoo

| Model | Dataset | mAP50 | mAP50-95 | Params(M) | FLOPs(G) | Weights |
|-------|---------|--------|----------|-----------|----------|---------|
| SCL-YOLO | TXL_PBC | 97.6% | 85.2% | 1.751 | 3.2 | [Download](link) |
| SCL-YOLO | CBC | 96.4% | 70.8% | 1.751 | 3.2 | [Download](link) |

## Results

### Performance Comparison

| Model | mAP50 | mAP50-95 | Params(M) | FLOPs(G) |
|-------|--------|----------|-----------|----------|
| YOLOv11n | 97.1% | 83.7% | 2.582 | 6.3 |
| **SCL-YOLO** | **97.6%** | **85.2%** | **1.751** | **3.2** |

### Ablation Study Results

| Components | mAP50 | mAP50-95 | Params(M) | FLOPs(G) |
|------------|--------|----------|-----------|----------|
| Baseline (YOLOv11n) | 97.1% | 83.7% | 2.582 | 6.3 |
| + StarNet | 97.1% | 83.6% | 1.943 | 5.0 |
| + CAPFFN | 97.4% | 85.0% | 1.987 | 4.2 |
| + LEDH (Full) | **97.6%** | **85.2%** | **1.751** | **3.2** |

## File Structure

```
SCL-YOLO/
├── configs/                 # Configuration files
│   ├── scl_yolo.yaml
│   └── datasets/
├── models/                  # Model implementations
│   ├── __init__.py
│   ├── scl_yolo.py
│   ├── backbone/
│   │   └── starnet.py
│   ├── neck/
│   │   └── capffn.py
│   └── head/
│       └── ledh.py
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── datasets.py
│   ├── general.py
│   ├── metrics.py
│   └── plots.py
├── weights/                 # Pre-trained weights
├── runs/                    # Training results
├── train.py                 # Training script
├── test.py                  # Testing script
├── detect.py                # Inference script
├── requirements.txt         # Dependencies
└── README.md               # This file
```


## Acknowledgments

- Thanks to the authors of YOLOv11 for the base architecture
- TXL_PBC dataset contributors for the high-quality blood cell dataset
- Ultralytics team for the excellent YOLO framework

## Contact

- Yang Fu - 429564385@qq.com
- Yong Hong Wu - whyflying2008@163.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Updates

- **2024-12**: Initial release of SCL-YOLO
- **2024-12**: Added support for CBC dataset
- **2024-12**: Released pre-trained weights

---

⭐ If you find this project helpful, please consider giving it a star!
