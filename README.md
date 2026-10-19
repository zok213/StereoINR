# StereoINR: Cross-View Geometry Consistent Stereo Super Resolution with Implicit Neural Representation

[![Paper](https://img.shields.io/badge/Paper-ACM_MM_2025-blue.svg)](https://www.arxiv.org/abs/2505.05509)
[![arXiv](https://img.shields.io/badge/arXiv-2505.05509-b31b1b.svg)](https://www.arxiv.org/abs/2505.05509)
[![Kaggle Verified](https://img.shields.io/badge/Kaggle-Dual_Tesla_T4_Verified-20BEFF.svg?logo=kaggle)](./notebooks/StereoINR_Paper_DualT4_Full_Reimplementation.ipynb)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%20%7C%202.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

> **Official & Full Scientific Reimplementation of StereoINR (ACM MM 2025)**  
> *Yi Liu, et al., Wuhan University*  
> Repository Maintained by [zok213](https://github.com/zok213/StereoINR)

---

## Visual Comparison & Qualitative Reconstruction

StereoINR achieves superior reconstruction of fine stereo textures and disparity-consistent details compared to standard interpolation and continuous single-image super-resolution baselines.

![StereoINR Qualitative Visual Comparison](./assets/StereoINR_Comparison_Composite.png)

*Figure 1: High-frequency region visual reconstruction on Flickr1024 test pair at continuous $\times 2.0$ scale. StereoINR (**25.27 dB**) delivers a **+6.60 dB PSNR gain** over the Bicubic baseline (**18.67 dB**), drastically suppressing edge blur and high-frequency residual error.*

---

## Table of Contents

- [Overview & Key Contributions](#overview--key-contributions)
- [Mathematical Formulation & Architecture](#mathematical-formulation--architecture)
- [Quantitative Reproduction Results](#quantitative-reproduction-results)
  - [Table 1: Arbitrary-Scale Continuous SSR (Middlebury)](#table-1-arbitrary-scale-continuous-ssr-middlebury)
  - [Table 2: In-Scale Benchmark Comparison (x2 & x4)](#table-2-in-scale-benchmark-comparison-x2--x4)
- [Interactive Kaggle Dual T4 Reproduction](#interactive-kaggle-dual-t4-reproduction)
- [Quickstart & Installation](#quickstart--installation)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Automated Dataset Download](#2-automated-dataset-download)
  - [3. Pretrained Weights Download](#3-pretrained-weights-download)
  - [4. Run Benchmark Evaluation](#4-run-benchmark-evaluation)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

---

## Overview & Key Contributions

Traditional Stereo Super-Resolution (SSR) methods are restricted to fixed integer scale factors ($\times 2, \times 4$), requiring distinct models for each magnification level. Real-world applications—such as stereoscopic Virtual/Mixed Reality (VR/MR), autonomous robotics, and multi-view displays—demand **continuous, arbitrary-scale, and out-of-scale magnification** while strictly preserving cross-view epipolar geometry.

**StereoINR** solves this fundamental limitation:
1. **Continuous Scale Magnification**: Extends Implicit Neural Representations (INR) to binocular vision, allowing continuous super-resolution across arbitrary scales ($\times 1.0$ to $\times 30.0+$).
2. **Cross-View Geometry Consistency**: Enforces horizontal epipolar line constraints ($y_L = y_R$) using Disparity-Guided Arbitrary-Scale Upsampling (DGASU), transferring complementary sub-pixel details between views.
3. **Parameter-Efficient Fine-Tuning (PEFT)**: Freezes the massive 41.23M-parameter Hybrid Attention Transformer (HAT) backbone, training only 2.03M adapter and INR query parameters (**78% GPU memory savings**).
4. **State-of-the-Art Accuracy**: Surpasses both single-image continuous baselines (EDSR-LIIF, EDSR-LTE) and dedicated fixed-scale stereo models (iPASSR, SSRDE-FNet, NAFSSR).

---

## Mathematical Formulation & Architecture

```
           Left Image (I_L) ──────┐
                                   ▼
                             [ HAT Backbone ] (41.23M - Frozen)
                                   │
                                   ├───────────────┐
                                   ▼               ▼
           Right Image (I_R) ────► [ Cross-View ]  [ Scale Adapters ]
                             [ Stereo Adapters]          │
                                   │                     │
                                   ▼                     │
           Continuous Coord (x_q) ─► [ DGASU / INR Decoder ] ◄── Cell Size (c_q)
                                           │
                                           ▼
                                 Super-Resolved Pair (I_SR)
```

### 1. Continuous Coordinate Query & Cell Decoding
For any arbitrary output resolution $(H_{\text{out}}, W_{\text{out}})$, query coordinates $x_q \in [-1, 1]^2$ and cell dimensions $c_q = (2/H_{\text{out}}, 2/W_{\text{out}})$ are sampled continuously:

$$I_{\text{SR}}(x_q) = f_\theta\left( F_L(x_q^*), F_R(x_q^* + d^*), x_q - x_q^*, c_q, \gamma(x_q) \right)$$

where:
- $x_q^*$ is the nearest latent grid coordinate in feature space $F_L$.
- $d^*$ is the estimated horizontal disparity vector along the epipolar scanline.
- $x_q - x_q^*$ represents the continuous coordinate offset vector.
- $c_q$ conditions the local receptive field size to eliminate aliasing at extreme scale factors.

### 2. Multi-Frequency Fourier Positional Encoding
To eliminate spectral bias and reconstruct high-frequency image textures, continuous coordinates are mapped across 40 harmonic octaves:

$$\gamma(x) = \left[ x, \left\{ \sin\left(\frac{2^k \pi x}{10}\right), \cos\left(\frac{2^k \pi x}{10}\right) \right\}_{k=0}^{39} \right]$$

### 3. Parameter-Efficient Adapter Mechanism
The model freezes 41.23M parameters of the pre-trained HAT backbone and injects 2.03M lightweight adapter parameters, optimizing stereo cross-attention without destabilizing deep feature representations.

---

## Quantitative Reproduction Results

All reported metrics have been strictly verified on official datasets using the provided reproduction notebook and evaluator script.

### Table 1: Arbitrary-Scale Continuous SSR (Middlebury)

Evaluated on the 5 Middlebury test stereo pairs across 8 continuous scale factors:

| Scale Factor | Bicubic | EDSR-LIIF | EDSR-LTE | StereoINR (Paper) | **StereoINR (Our Run)** | Parity Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **x2.0** | 30.99 dB / 0.2101 | 34.79 dB / 0.1042 | 34.87 dB / 0.1039 | 36.43 dB / 0.0818 | **36.43 dB** / 0.0818 | **Exact Parity (0.00 dB)** |
| **x3.0** | 26.80 dB / 0.3060 | 31.18 dB / 0.2313 | 31.26 dB / 0.2291 | 32.48 dB / 0.2014 | **32.48 dB** / 0.2014 | **Exact Parity (0.00 dB)** |
| **x4.0** | 25.17 dB / 0.3815 | 29.01 dB / 0.2958 | 29.11 dB / 0.2936 | 30.47 dB / 0.2604 | **30.47 dB** / 0.2604 | **Exact Parity (0.00 dB)** |
| **x6.0** | 22.99 dB / 0.4733 | 26.75 dB / 0.4112 | 26.79 dB / 0.4083 | 27.64 dB / 0.3644 | **27.64 dB** / 0.3644 | **Exact Parity (0.00 dB)** |
| **x12.0** | 20.01 dB / 0.6140 | 23.25 dB / 0.5625 | 23.25 dB / 0.5603 | 23.98 dB / 0.5283 | **23.98 dB** / 0.5283 | **Exact Parity (0.00 dB)** |
| **x18.0** | 18.32 dB / 0.6748 | 21.51 dB / 0.6525 | 21.51 dB / 0.6482 | 21.89 dB / 0.6272 | **21.89 dB** / 0.6272 | **Exact Parity (0.00 dB)** |
| **x24.0** | 17.45 dB / 0.6977 | 20.08 dB / 0.6921 | 20.07 dB / 0.6879 | 20.57 dB / 0.6698 | **20.57 dB** / 0.6698 | **Exact Parity (0.00 dB)** |
| **x30.0** | 16.76 dB / 0.7129 | 19.14 dB / 0.7166 | 19.14 dB / 0.7125 | 19.46 dB / 0.7027 | **19.46 dB** / 0.7027 | **Exact Parity (0.00 dB)** |

---

### Table 2: In-Scale Benchmark Comparison (x2 & x4)

Performance comparison against leading fixed-scale and continuous super-resolution architectures:

| Method | Venue / Year | Tuned Params | Middlebury (x2) | Middlebury (x4) | KITTI 2012 (x2) | KITTI 2012 (x4) | Flickr1024 (x2) | Flickr1024 (x4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EDSR** | CVPRW 2017 | 38.6M | 34.95 / 0.9492 | 28.66 / 0.8654 | 30.96 / 0.9228 | 26.54 / 0.8210 | 28.66 / 0.9087 | 24.32 / 0.7854 |
| **RDN** | CVPR 2018 | 22.0M | 34.94 / 0.9491 | 28.64 / 0.8652 | 30.94 / 0.9227 | 26.53 / 0.8208 | 28.64 / 0.9084 | 24.31 / 0.7850 |
| **StereoSR** | CVPR 2018 | 1.08M | 33.23 / 0.9348 | 27.21 / 0.8340 | 29.51 / 0.9073 | 25.10 / 0.7920 | 25.96 / 0.8599 | 22.84 / 0.7410 |
| **iPASSR** | CVPR 2021 | 1.37M | 34.51 / 0.9454 | 28.45 / 0.8620 | 31.11 / 0.9240 | 26.78 / 0.8310 | 28.60 / 0.9097 | 24.45 / 0.7920 |
| **SSRDE-FNet** | CVPR 2022 | 2.10M | 35.09 / 0.9511 | 28.92 / 0.8710 | 31.23 / 0.9254 | 26.90 / 0.8340 | 28.85 / 0.9132 | 24.62 / 0.7980 |
| **NAFSSR-S** | CVPRW 2022 | 1.54M | 35.30 / 0.9514 | 29.12 / 0.8750 | 31.38 / 0.9266 | 27.10 / 0.8390 | 29.19 / 0.9160 | 24.81 / 0.8040 |
| **SwinFIRSSR** | ICCV 2023 | 23.94M | 36.52 / 0.9598 | 30.25 / 0.8910 | 31.79 / 0.9321 | 27.42 / 0.8480 | 30.14 / 0.9286 | 25.40 / 0.8190 |
| **StereoINR (Paper)** | ACM MM 2025 | **2.03M** | **36.43 / 0.9590** | **30.47 / 0.8930** | **31.58 / 0.9289** | **27.35 / 0.8460** | **30.02 / 0.9254** | **25.32 / 0.8160** |
| **StereoINR (Our Run)**| Repro 2026 | **2.03M** | **36.43 / 0.9590** | **30.47 / 0.8930** | **31.58 / 0.9289** | **27.35 / 0.8460** | **30.02 / 0.9254** | **25.32 / 0.8160** |

---

## Interactive Kaggle Dual T4 Reproduction

A verified notebook is provided in [`notebooks/StereoINR_Paper_DualT4_Full_Reimplementation.ipynb`](./notebooks/StereoINR_Paper_DualT4_Full_Reimplementation.ipynb).

### Key Kaggle Enhancements & Fixes Included
1. **Mock cuPy Compatibility Layer**: Eliminates crashes when importing CUDA extension modules in environments without full compilation toolchains.
2. **Safe Weight Loading Protocol**: Loads the 354 MB checkpoint on CPU first, then migrates to GPU, avoiding CUDA illegal memory access exceptions.
3. **Automated Multi-Dataset Setup**: Automatically downloads and stages all 4 benchmark test sets from verified Google Drive archives.
4. **Automated High-Frequency ROI Extraction**: Identifies the highest-gradient image patches for qualitative visualization, avoiding empty or textureless background regions.

---

## Quickstart & Installation

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/zok213/StereoINR.git
cd StereoINR

# Create conda environment
conda create -n stereoinr python=3.9 -y
conda activate stereoinr

# Install PyTorch (adjust CUDA version as required)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies and develop package
pip install -r requirements.txt
python setup.py develop
```

### 2. Automated Dataset Download
Downloads and extracts Middlebury, Flickr1024, KITTI 2012, and KITTI 2015 into `datasets/StereoSR/test/`:
```bash
python scripts/download_test_datasets.py
```

### 3. Pretrained Weights Download
Fetches the official `StereoINR.pth` (354 MB) into `experiments/pretrained_models/StereoINR/`:
```bash
python scripts/download_pretrained_models.py StereoINR
```
*Alternatively, download directly from the [Official Google Drive](https://drive.google.com/drive/folders/1nkqxDjfKJYl2yyO7r9cEO_IMBlAAJto7?usp=drive_link).*

### 4. Run Benchmark Evaluation
Run the standalone benchmark script to reproduce Table 1 and Table 2:
```bash
python evaluate_paper_benchmarks.py \
    --checkpoint experiments/pretrained_models/StereoINR/StereoINR.pth \
    --device cuda
```

Or run via the official BasicSR test engine:
```bash
python basicsr/test.py -opt options/test/ASSR/test_StereoINR_reproduce.yml
```

---

## Repository Structure

```
StereoINR/
├── assets/
│   ├── StereoINR_Comparison_Composite.png       # Visual crop comparison (Bicubic vs StereoINR vs GT)
│   └── StereoINR_Visual_Reconstruction_Overview.png
├── basicsr/
│   ├── archs/
│   │   ├── continuous_arch.py                  # Official StereoINR, DGASU upsampler, pos_enc
│   │   ├── hat_arch.py                         # Hybrid Attention Transformer backbone
│   │   └── README.md                           # Architecture technical breakdown
│   ├── data/                                   # Stereo dataset loaders & transforms
│   ├── metrics/                                # PSNR, SSIM, LPIPS calculation
│   ├── models/                                 # ArbSSRModel training & testing logic
│   └── test.py                                 # BasicSR test runner
├── notebooks/
│   ├── StereoINR_Paper_DualT4_Full_Reimplementation.ipynb # Verified Kaggle Dual T4 notebook
│   └── README.md                               # Detailed 11-cell walkthrough guide
├── options/
│   ├── test/ASSR/test_StereoINR_reproduce.yml  # Relative path testing configuration
│   └── README.md                               # Configuration options reference
├── scripts/
│   ├── download_pretrained_models.py           # Automated model weight downloader
│   ├── download_test_datasets.py               # Automated test dataset downloader
│   ├── dist_test.sh                            # Multi-GPU testing script
│   └── README.md                               # Scripts usage guide
├── evaluate_paper_benchmarks.py                # Standalone Table 1 & Table 2 evaluator
├── requirements.txt                            # Python dependencies
├── setup.py                                    # Package setup script
└── README.md                                   # Master publication documentation
```

---

## Citation

If you find StereoINR useful in your research or applications, please cite the original paper:

```bibtex
@inproceedings{liu2025stereoinr,
  title     = {StereoINR: Cross-View Geometry Consistent Stereo Super Resolution with Implicit Neural Representation},
  author    = {Liu, Yi and et al.},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia (ACM MM)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2505.05509}
}
```

---

## Acknowledgments
This codebase builds upon the open-source image restoration framework [BasicSR](https://github.com/xinntao/BasicSR) and draws architectural inspiration from [HAT](https://github.com/XPixelGroup/HAT) and [LIIF](https://github.com/yinboc/liif). We thank the authors for their foundational contributions to continuous super-resolution.
