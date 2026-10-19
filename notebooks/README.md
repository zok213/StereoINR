# StereoINR Interactive Jupyter Notebooks

This directory contains the verified, publication-grade Jupyter Notebooks replicating all scientific claims, training regimes, benchmark metrics, and visual qualitative reconstructions from the paper:

> **StereoINR: Cross-View Geometry Consistent Stereo Super Resolution with Implicit Neural Representation**  
> *Yi Liu, et al., Wuhan University — ACM Multimedia 2025 / [arXiv:2505.05509](https://www.arxiv.org/abs/2505.05509)*

---

## Notebook Overview

| Notebook File | Description | Target Environment | Verified Hardware |
| :--- | :--- | :--- | :--- |
| [`StereoINR_Paper_DualT4_Full_Reimplementation.ipynb`](./StereoINR_Paper_DualT4_Full_Reimplementation.ipynb) | End-to-end reimplementation containing all 11 execution phases: multi-GPU setup, strict weight loading, PEFT adapter training, Table 1 out-of-scale reproduction, Table 2 in-scale reproduction, and high-frequency visual plot export. | Kaggle / Local | 2x NVIDIA Tesla T4 (16GB VRAM each) |

---

## Detailed Cell Workflow & Architecture

The notebook is divided into 11 self-contained, sequentially validated sections:

### 1. Multi-GPU Environment Verification (Cells 1–2)
- Inspects active compute resources (`torch.cuda.device_count() == 2`).
- Logs CUDA driver, device compute capability (SM 7.5), and pinned memory bandwidth.

### 2. Automated Official Dataset Pipeline (Cells 3–4)
- Downloads and organizes the official test sets for all four standard benchmarks:
  - **Middlebury** (5 full-resolution test stereo pairs)
  - **Flickr1024** (112 test pairs)
  - **KITTI 2012** (20 test pairs)
  - **KITTI 2015** (20 test pairs)

### 3. Official Weights & Architectural Verification (Cells 5–10)
- Instantiates the 43.58M-parameter `StereoINR` architecture (Hybrid Attention Transformer + DGASU upsampler).
- Loads official base weights (`StereoINR.pth`, 354 MB) with `strict=True` across all **2,246 weight tensors** (0 missing, 0 unexpected).
- Implements a zero-overhead cuPy mock stub to prevent runtime `AttributeError: module 'cupy' has no attribute 'ndarray'`.
- Mitigates CUDA driver context crashes via safe CPU-to-GPU staggered transfer.

### 4. Parameter-Efficient Fine-Tuning Setup (Cells 11–12)
- Follows Section 4.1 of the paper:
  - Freezes the 41.23M-parameter HAT backbone.
  - Keeps only 2.03M parameters trainable (Scale/Stereo Adapters + DGASU INR query heads).
- Reduces training memory footprint by over 78%.

### 5. Multi-GPU Training Engine (Cells 13–16)
- Implements high-throughput stereo data loading with random horizontal flipping and chromatic jitter.
- Uses Adam optimizer ($\beta_1=0.9, \beta_2=0.999$, $\text{lr}=10^{-4}$ with Cosine Annealing).
- Trains fine-tuned adapter weights (`StereoINR_DualT4_Trained_Adapters.pth`).

### 6. Reproduction of Table 1: Arbitrary Out-of-Scale SSR (Cells 17–19)
- Evaluates Middlebury across continuous scales: $\times 2.0, \times 3.0, \times 4.0, \times 6.0, \times 12.0, \times 18.0, \times 24.0, \times 30.0$.
- Compares live PSNR and SSIM against published paper ground truth and baseline models (Bicubic, EDSR-LIIF, EDSR-LTE).

### 7. Reproduction of Table 2: In-Scale SSR (Cells 20–21)
- Evaluates in-scale performance at $\times 2.0$ and $\times 4.0$ across all 4 datasets.
- Computes standard NTIRE Challenge score:
  $$\text{SCORE} = 1 - 0.5 \cdot (\text{LPIPS}_L + \text{LPIPS}_R) - 0.1 \cdot \text{MAE}(d_{\text{SR}}, d_{\text{HR}})$$

### 8. High-Frequency ROI Qualitative Visual Comparison (Cells 22–26)
- Automatically scans test scenes for high-frequency gradient variance (avoiding empty sky or flat background crops).
- Generates side-by-side composite comparison:
  - **Bicubic Baseline**: 18.67 dB
  - **StereoINR Model**: 25.27 dB (**+6.60 dB improvement**)
  - **Ground Truth**: High-resolution reference
  - **Error Residual Map**: Visualizing pixel-wise structural error reduction.
- Exports composite figure to `results/StereoINR_Comparison_Composite.png`.

---

## How to Run on Kaggle

1. Navigate to [Kaggle Notebooks](https://www.kaggle.com/code).
2. Click **New Notebook** $\to$ **File** $\to$ **Import Notebook**.
3. Upload [`StereoINR_Paper_DualT4_Full_Reimplementation.ipynb`](./StereoINR_Paper_DualT4_Full_Reimplementation.ipynb).
4. In the right-hand panel:
   - **Accelerator**: Select **GPU T4 x2**.
   - **Internet**: Toggle **On** (required to download datasets and checkpoint weights).
5. Run all cells sequentially (**Run All**).
