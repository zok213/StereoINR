# IEEE Conference Paper: Scale-Liberated StereoINR

This directory contains the complete, publication-ready academic paper formatted according to standard **IEEE Conference / Transactions** specifications.

## Files
- [`main.tex`](main.tex): The master LaTeX source file containing 9 comprehensive sections, mathematical formulations, LaTeX tables, and figure environments.
- [`references.bib`](references.bib): BibTeX bibliography database with citations for SOTA INR, StereoSR, and XR hardware architectures.
- Associated figures are referenced directly from [`outputs/presentation_slides/`](../../outputs/presentation_slides/).

---

## How to Compile

### Option 1: Using Overleaf (Recommended)
1. Zip this `reports/ieee_paper/` folder along with the referenced figures in `outputs/presentation_slides/`.
2. Upload the `.zip` archive to [Overleaf](https://www.overleaf.com/).
3. Set the compiler to `pdfLaTeX` (TeX Live 2024+).
4. Click **Recompile**.

### Option 2: Local Command Line (Linux / macOS / Windows with TeX Live or MiKTeX)
```bash
cd reports/ieee_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Paper Outline & Core Technical Contributions
1. **Section I: Introduction** — The passthrough resolution crisis in consumer VR headsets (Meta Quest 3S, 20 PPD limit, 11.11 ms 90 FPS budget).
2. **Section II: Related Work** — Implicit Neural Representations (LIIF, LTE) and Stereo SR (PASSRNet, NAFSSR, StereoINR).
3. **Section III: System Architecture & Mathematical Formulation** — Rectified epipolar geometry ($y_L = y_R$), 1D Epipolar Parallax Attention (EPA-1D), Lipschitz-bounded Dynamic Tanh (DyT), and Continuous Fourier H-LTE Decoder.
4. **Section IV: Multi-Task Loss Formulation & Distillation** — Charbonnier Loss, Epipolar Disparity Loss, 2D FFT Frequency Loss, and Knowledge Distillation ($\mathcal{L}_{KD}$).
5. **Section V: Experimental Benchmark Setup** — 8-phase closed-loop Meta Quest 3S Harvested dataset and international benchmarks (Middlebury, KITTI 2012, KITTI 2015, Flickr1024).
6. **Section VI: Quantitative & Qualitative Evaluation** — 15-scene Crop PSNR breakdown, Stereo MAE disparity consistency evaluation, and mathematical resolution of the Retinal Rivalry paradox.
7. **Section VII: Hardware Profiling & Embedded Deployment** — Dual Tesla T4 DDP benchmarking, Qualcomm AI Hub physical execution on Snapdragon 8 Gen 2 / Hexagon V73 NPU, and INT8 numerical parity validation.
8. **Section VIII: Discussion, Limitations, & Engineering Roadmap** — The 2.58 dB PSNR trade-off, Meta Horizon OS SELinux sandbox lockout, and Foveated Passthrough implementation (~4.2 ms per 256x256 tile).
9. **Section IX: Conclusion** — Summary and future directions for XR passthrough neural supersampling.
