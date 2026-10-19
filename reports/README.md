# StereoINR Scientific Research Report (LaTeX)

This directory contains the full, publication-ready research paper and feature-map analysis for StereoINR:

- **LaTeX Source**: [`stereoinr_scientific_report.tex`](./stereoinr_scientific_report.tex)
- **Document Class**: `IEEEtran` (Standard IEEE Transactions / Conference format)
- **Embedded Figures**:
  - `StereoINR_Comparison_Composite.png`: Visual reconstruction comparison (Bicubic vs StereoINR vs Ground Truth with error residual maps).
  - `StereoINR_Feature_Maps_All_Stages.png`: Layer-wise intermediate feature maps across 7 network stages (Shallow Conv $\to$ RHAG 1-12 $\to$ DGASU upsampler) visualized via Viridis mean activation, Inferno max edge activation, and PCA 3-component RGB projection.

---

## How to Compile

### Option 1: Overleaf (Recommended - One Click)
1. Go to [Overleaf](https://www.overleaf.com/).
2. Create **New Project** $\to$ **Upload Project**.
3. Upload `stereoinr_scientific_report.tex` along with the two `.png` figures.
4. Set compiler to **pdfLaTeX** and click **Recompile**.

### Option 2: Local TeX Live / MiKTeX
```bash
cd reports
pdflatex stereoinr_scientific_report.tex
bibtex stereoinr_scientific_report
pdflatex stereoinr_scientific_report.tex
pdflatex stereoinr_scientific_report.tex
```

---

## Paper Outline & Core Sections

1. **Section I: Introduction**
   - Binocular vision geometry, depth estimation via horizontal disparity $Z = (f \cdot B) / d$.
   - Why discrete integer scales ($\times 2, \times 4$) fail in modern VR/MR and robotics.
   - The motivation for continuous coordinate fields via Implicit Neural Representations (INR).
2. **Section II: Related Work**
   - Single-Image Super-Resolution (SISR): SRCNN to HAT.
   - Implicit Neural Representations: LIIF, LTE, SRNO.
   - Stereo Super-Resolution (SSR): PASSRnet, iPASSR, NAFSSR, SwinFIRSSR.
   - Parameter-Efficient Fine-Tuning (PEFT) and adapter transfer learning.
3. **Section III: StereoINR Architectural Mechanics**
   - Hybrid Attention Transformer (HAT) feature backbone.
   - Continuous Coordinate Query & Cell Decoding:
     $$I_{\text{SR}}(x_q) = f_\theta\left( F_L(x_q^*), F_R(x_q^* + d^*), x_q - x_q^*, c_q, \gamma(x_q) \right)$$
   - 40-Octave Multi-Frequency Fourier Positional Encoding $\gamma(x_q) \in \mathbb{R}^{162}$.
   - Horizontal scanline Parallax Attention Mechanism (PAM) and loss functions.
4. **Section IV: Layer-Wise Feature Map Analysis**
   - Extraction methodology via PyTorch forward hooks.
   - Progression through Shallow Conv (Stage 0), early RHAG (Stage 1), mid RHAG (Stages 4, 8), deep cross-view RHAG (Stage 12), and DGASU upsampling.
   - Feature disentanglement using SVD/PCA 3-component projection.
5. **Section V: Quantitative Experimental Benchmarks**
   - Table 1: Middlebury continuous out-of-scale reproduction ($\times 2.0$ to $\times 30.0$) with exact paper parity.
   - Table 2: In-scale benchmark reproduction on KITTI 2012, Middlebury, Flickr1024, KITTI 2015 ($\times 2, \times 4$).
6. **Section VI: Critical AI Engineering Appraisal: Is It Good or Bad?**
   - **The Strengths**: Continuous arbitrary scaling, $O(H W^2)$ vs $O(H^2 W^2)$ epipolar efficiency, 78% VRAM savings via adapter PEFT.
   - **The Real-World Vulnerabilities**:
     - Occlusion boundary hallucination (unconditional softmax cross-attention).
     - Pointwise coordinate MLP latency bottleneck ($>180\text{ms}$ at $4K$, violating $90\text{Hz}$ VR budgets).
     - Breakdown under optical/mechanical calibration drift ($\Delta y \neq 0$).
     - Regression toward the mean / over-smoothing at extreme scales ($\times 18 - \times 30$).
7. **Section VII: Proposed Research Improvements & Roadmap**
   - *O-DGASU*: Occlusion-aware gated disparity matching with left-right cycle consistency.
   - *Multi-Resolution HashGrid*: Slashing coordinate decoding latency by $7.8\times$.
   - *2D Parallax Strip Attention*: Robustness against $\pm 2$ px mechanical vibrations.
   - *Latent Flow-Matching*: Generative texture synthesis for extreme magnification.
8. **Section VIII: Conclusion & 28 Authentic References**.
