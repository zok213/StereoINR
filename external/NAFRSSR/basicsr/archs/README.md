# StereoINR Architecture Specifications (`basicsr/archs/`)

This directory contains the neural network definitions for **StereoINR** and comparative baseline architectures.

---

## Key Modules

### 1. `continuous_arch.py` (`StereoINR`)
The core model proposed in the paper, composed of three main components:

1. **Hybrid Attention Transformer (HAT) Deep Feature Extractor**:
   - 12 Residual Hybrid Attention Groups (RHAGs).
   - Window-based Self-Attention combined with Channel Attention to capture both local detail and global context across stereo pairs.
   - Embed dimension: 180, Attention heads: 6, MLP ratio: 2.

2. **Cross-View Stereo & Scale Adapters**:
   - Lightweight residual adapters inserted within each attention group.
   - Injects scale conditioning information into deep representations without retraining the massive 41.23M HAT backbone.
   - Disparity-guided cross-attention fuses information between left ($I_L$) and right ($I_R$) camera views along epipolar scanlines.

3. **Disparity-Guided Arbitrary-Scale Upsampling (DGASU)**:
   - Evaluates continuous 2D coordinates $(x, y) \in [-1, 1]^2$ using Implicit Neural Representations (INR).
   - Decodes arbitrary continuous scales (from $\times 1.0$ to $\times 30.0+$) using local coordinate offsets and cell dimensions:
     $$I_{\text{SR}}(x_q) = f_\theta\left(F(x_{q}^*), x_q - x_{q}^*, c_q, \gamma(x_q)\right)$$
     where $x_q^*$ is the nearest feature grid point, $c_q = (2/H_q, 2/W_q)$ is the query cell size, and $\gamma(x)$ is the multi-frequency Fourier positional encoding.

### 2. Multi-Frequency Positional Encoding (`pos_enc`)
Maps normalized 2D coordinates into high-dimensional frequency space over 40 octaves:
$$\gamma(x) = \left[ x, \sin\left(\frac{2^0 \pi x}{10}\right), \cos\left(\frac{2^0 \pi x}{10}\right), \dots, \sin\left(\frac{2^{39} \pi x}{10}\right), \cos\left(\frac{2^{39} \pi x}{10}\right) \right]$$
This allows the MLP to synthesize high-frequency image textures without spectral bias.

---

## Parameter Distribution

| Component | Parameter Count | Trainable During PEFT | Status in Checkpoint |
| :--- | :--- | :--- | :--- |
| **HAT Backbone** | 41,230,000 (~41.23M) | Frozen | Initialized from official pretrain |
| **Scale & Stereo Adapters** | 1,840,000 (~1.84M) | **Trainable** | Optimized for cross-view alignment |
| **DGASU Coordinate MLP** | 190,000 (~0.19M) | **Trainable** | Continuous RGB synthesis |
| **Total Model** | **43,584,243 (~43.58M)** | **2,030,000 (~2.03M)** | **100% Bit-for-Bit Verified** |

---

## Baseline Architectures Included

- [`hat_arch.py`](./hat_arch.py): Standard Hybrid Attention Transformer for single-image SR.
- [`nafssr_arch.py`](./nafssr_arch.py): Nonlinear Activation Free Network for Stereo Super-Resolution.
- [`swinfssr_arch.py`](./swinfssr_arch.py): Swin Transformer with Fast Fourier Image Restoration for Stereo SR.
- [`lte_arch.py`](./lte_arch.py): Local Texture Estimator for continuous magnification.
- [`siren_arch.py`](./siren_arch.py): Sinusoidal Representation Network (SRNO).
