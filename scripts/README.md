# StereoINR Automation Scripts (`scripts/`)

This directory contains utility scripts for downloading datasets and pretrained weights, distributed training/testing, and weight management.

---

## Quick Usage

### 1. Download Official Pretrained Model Checkpoint
Downloads the official 43.58M-parameter `StereoINR.pth` (354 MB) into `experiments/pretrained_models/StereoINR/`:
```bash
python scripts/download_pretrained_models.py StereoINR
```

To download all comparative baselines (EDSR, ESRGAN, etc.):
```bash
python scripts/download_pretrained_models.py all
```

---

### 2. Download Benchmark Test Datasets
Downloads and unpacks all 4 official test sets (~490 MB archive) into `datasets/StereoSR/test/`:
```bash
python scripts/download_test_datasets.py
```
This prepares:
- `datasets/StereoSR/test/Middlebury/hr` (5 pairs)
- `datasets/StereoSR/test/Flickr1024/hr` (112 pairs)
- `datasets/StereoSR/test/KITTI2012/hr` (20 pairs)
- `datasets/StereoSR/test/KITTI2015/hr` (20 pairs)

---

### 3. Distributed Multi-GPU Execution
Launch distributed testing across multiple GPUs using PyTorch DDP:
```bash
bash scripts/dist_test.sh 2 options/test/ASSR/test_StereoINR_reproduce.yml
```
*(where `2` is the number of GPUs to use)*.
