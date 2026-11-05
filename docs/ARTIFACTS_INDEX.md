# EdgeStereoINR: Central Artifacts & Knowledge Architecture Index

This index provides a complete, authoritative mapping of all technical artifacts, checkpoints, empirical benchmarks, and deployment packages produced for the **EdgeStereoINR** project on **Meta Quest 3S (Snapdragon XR2 Gen 2)**.

---

## 1. Authoritative Core Knowledge Bases

| Document | Path | Scope & Purpose |
| :--- | :--- | :--- |
| **Project Master Memory** | [`PROJECT_MEMORY.md`](file:///d:/Gitrepo/Active%20Stereo%20RL/PROJECT_MEMORY.md) | The root engineering blueprint: Hardware Truth Matrix (Host PC vs Quest 3S), FastRPC SELinux lockout explanation, HAT vs StereoINR vs DyT lineage, bug fix ledger, and 157-pair benchmark table. |
| **Vietnamese Technical Handbook** | [`docs/PROJECT_KNOWLEDGE_BASE_VN.md`](file:///d:/Gitrepo/Active%20Stereo%20RL/docs/PROJECT_KNOWLEDGE_BASE_VN.md) | Comprehensive engineering guide in Vietnamese covering firmware sandbox constraints, Adreno 740 SRAM budget, mathematical parity proofs, and Unity Sentis deployment runbooks. |
| **Master Dissertation Artifact** | [`artifact_dai_luan_an_toan_dien_edgestreoinr_quest3s_final.md`](file:///C:/Users/Admin/.gemini/antigravity/brain/0d24f9ec-9bf1-44ec-8eeb-1a1952048fb8/artifact_dai_luan_an_toan_dien_edgestreoinr_quest3s_final.md) | Exhaustive 10-chapter technical dissertation detailing theoretical formulations, mathematical proofs, ablation studies, and comparative analysis against CVPR/ECCV literature. |
| **Multi-Model Critique Artifact** | [`artifact_dai_luan_an_phan_bien_da_chieu_tat_ca_cac_model_stereo_sr_quest3s.md`](file:///C:/Users/Admin/.gemini/antigravity/brain/0d24f9ec-9bf1-44ec-8eeb-1a1952048fb8/artifact_dai_luan_an_phan_bien_da_chieu_tat_ca_cac_model_stereo_sr_quest3s.md) | Authoritative comparative analysis of all 11 models (2016-2026), Adreno 740 physical GMEM bounds, and RepDyT structural reparameterization derivations. |

---

## 2. Checkpoint & Weight Lineage (`weights/`)

| Version | File | Size | Architecture & Role |
| :--- | :--- | :--- | :--- |
| **v5** | `weights/kaggle_dual_t4/dyt_nafrssr_quest3s_best.pth` | 403 KB | Baseline DyT-NAFSSR with Dynamic Tanh (width 48, 4 blocks). |
| **v6** | `weights/kaggle_dual_t4/dyt_nafrssr_quest3s_v6_best.pth` | 403 KB | Interleaved Recurrent Cross-Attention (IRCA) with shared WindowedSCAM. |
| **v8** | `weights/kaggle_dual_t4/dyt_continuous_inr_quest3s_best.pth` | 495 KB | Continuous scale adapter conditioned on $r \in [1.2, 4.0]$. |
| **v9** | `weights/kaggle_dual_t4/dyt_continuous_inr_quest3s_v9_best.pth` | 508 KB | Teacher distillation hooks with Inter-Feature Distillation (IFD). |
| **v10 Notebook** | [`deployment/kaggle_dual_t4_training/train_quest3s_v10_repdyt_dual_t4.ipynb`](file:///d:/Gitrepo/Active%20Stereo%20RL/deployment/kaggle_dual_t4_training/train_quest3s_v10_repdyt_dual_t4.ipynb) | 691 KB | Version 10 RepDyT Dual Tesla T4 training notebook with IFD and Pareto warm-start. |
| **RepDyT Backbone** | [`src/edge_stereoinr/phase2_edge_inr/backbones/rep_dyt_continuous_inr.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/src/edge_stereoinr/phase2_edge_inr/backbones/rep_dyt_continuous_inr.py) | 5.2 KB | Structural Reparameterization Continuous Stereo INR (117K deploy params, 0.00e+00 error). |
| **Soup (Best)** | **`weights/kaggle_dual_t4/dyt_continuous_inr_quest3s_soup_best.pth`** | **491 KB** | **Pareto Model Soup ($0.5\theta_{v8} + 0.3\theta_{v6} + 0.2\theta_{v9}$) — 104,435 parameters.** |
| **Sentis ONNX** | **`weights/kaggle_dual_t4/dyt_continuous_inr_quest3s_soup_fp16.onnx`** | **1.49 MB** | **Production Unity 6 Sentis model for Adreno 740 GPU (Vulkan Compute).** |
| **QNN Static ONNX** | `weights/qnn_npu/dyt_edge_inr_qnn_tile64_4x.onnx` | 1.43 MB | Static $64\times 64 \to 256\times 256$ tile model for Qualcomm QNN SDK. |
| **QNN INT8 ONNX** | `weights/qnn_npu/dyt_edge_inr_qnn_tile64_4x_int8.onnx` | 1.66 MB | Quantized INT8 model achieving $57.1\text{ dB}$ SNR parity vs FP32. |

---

## 3. Deployment Engines & Shaders (`deployment/`)

| Component | Path | Description |
| :--- | :--- | :--- |
| **Unity Sentis Package** | [`deployment/unity_sentis/dist/com.activestereo.edgestreoinr-1.0.0.tgz`](file:///d:/Gitrepo/Active%20Stereo%20RL/deployment/unity_sentis/dist/com.activestereo.edgestreoinr-1.0.0.tgz) | Ready-to-import Unity UPM tarball for Unity 6 LTS. |
| **Tiled Streamer** | [`deployment/unity_sentis/TiledStereoSentisStreamer.cs`](file:///d:/Gitrepo/Active%20Stereo%20RL/deployment/unity_sentis/TiledStereoSentisStreamer.cs) | GPU-resident tiled streaming engine with auto-tensor name detection and zero CPU readback. |
| **Foveated Streamer** | [`deployment/unity_sentis/FoveatedStereoSentisStreamer.cs`](file:///d:/Gitrepo/Active%20Stereo%20RL/deployment/unity_sentis/FoveatedStereoSentisStreamer.cs) | Foveated super-resolution (FFSR) reducing compute by 45-60% via eccentricity scaling. |
| **Interactive Demo** | [`deployment/unity_sentis/StereoPassthroughDemo.cs`](file:///d:/Gitrepo/Active%20Stereo%20RL/deployment/unity_sentis/StereoPassthroughDemo.cs) | Turnkey test harness with synthetic stereo generator and live telemetry HUD. |
| **Hann Blend Shader** | [`deployment/unity_sentis/HannBlendShader.shader`](file:///d:/Gitrepo/Active%20Stereo%20RL/deployment/unity_sentis/HannBlendShader.shader) | Additive GPU blending shader applying 2D Hann-window cosine feathering. |
| **Hann Resolve Shader** | [`deployment/unity_sentis/HannResolveShader.shader`](file:///d:/Gitrepo/Active%20Stereo%20RL/deployment/unity_sentis/HannResolveShader.shader) | Normalization pass dividing accumulated RGB by weight sum to eliminate edge vignetting. |
| **Temporal Blend Shader** | [`deployment/unity_sentis/TemporalBlendShader.shader`](file:///d:/Gitrepo/Active%20Stereo%20RL/deployment/unity_sentis/TemporalBlendShader.shader) | Motion-warped temporal hysteresis shader eliminating 90.9% of inter-frame passthrough scintillation. |
| **Temporal Streamer** | [`deployment/unity_sentis/TemporalStereoSentisStreamer.cs`](file:///d:/Gitrepo/Active%20Stereo%20RL/deployment/unity_sentis/TemporalStereoSentisStreamer.cs) | C# streamer incorporating motion-compensated temporal feedback into live VR camera pipeline. |
| **C++ QNN Bridge** | [`deployment/android/qnn_native_bridge.cpp`](file:///d:/Gitrepo/Active%20Stereo%20RL/deployment/android/qnn_native_bridge.cpp) | Native C++ bridge for Qualcomm QNN SDK v2.18+ C-API (OEM / privileged apps). |
| **Vulkan Compute** | [`deployment/vulkan/edge_stereoinr.comp`](file:///d:/Gitrepo/Active%20Stereo%20RL/deployment/vulkan/edge_stereoinr.comp) | Catmull-Rom sub-pixel interpolation compute shader for low-level Vulkan fallback. |

---

## 4. Empirical Evaluation & Benchmark Reports (`logs/` & `scripts/`)

| Report / Script | Path | Coverage & Findings |
| :--- | :--- | :--- |
| **Exhaustive 157-Pair Report** | [`logs/benchmarks/canonical_157_pairs_report.json`](file:///d:/Gitrepo/Active%20Stereo%20RL/logs/benchmarks/canonical_157_pairs_report.json) | Complete JSON telemetry for all 157 pairs across Middlebury, KITTI 2012, KITTI 2015, and Flickr1024. |
| **Middlebury Head-to-Head Report** | [`logs/benchmarks/empirical_middlebury_head_to_head.json`](file:///d:/Gitrepo/Active%20Stereo%20RL/logs/benchmarks/empirical_middlebury_head_to_head.json) | Head-to-head empirical metrics comparing Bicubic, DyT-Edge-INR, NAFRSSR-M, and NAFRSSR-T. |
| **Temporal Stability Report** | [`logs/benchmarks/temporal_stability_benchmark.json`](file:///d:/Gitrepo/Active%20Stereo%20RL/logs/benchmarks/temporal_stability_benchmark.json) | Verified 90.9% temporal jitter reduction in 90 FPS VR passthrough simulation. |
| **Precision & Quantization Audit** | [`logs/benchmarks/precision_quantization_audit.json`](file:///d:/Gitrepo/Active%20Stereo%20RL/logs/benchmarks/precision_quantization_audit.json) | Empirical comparison of FP32 vs Native FP16 (73.81 dB PSNR) vs INT8 (47.81 dB SNR) on Adreno 740. |
| **Vertical Parallax Tolerance Report** | [`logs/benchmarks/epipolar_vertical_tolerance.json`](file:///d:/Gitrepo/Active%20Stereo%20RL/logs/benchmarks/epipolar_vertical_tolerance.json) | Evaluates epipolar cross-attention robustness under simulated rolling shutter skew up to 2.0 px (< 0.02 dB drop). |
| **Radiometric Tolerance Report** | [`logs/benchmarks/radiometric_exposure_tolerance.json`](file:///d:/Gitrepo/Active%20Stereo%20RL/logs/benchmarks/radiometric_exposure_tolerance.json) | Evaluates cross-attention stability across dual-camera exposure mismatches in [0.80x, 1.20x] (delta <= 0.01 dB). |
| **Occlusion Gating Report** | [`logs/benchmarks/occlusion_gating_benchmark.json`](file:///d:/Gitrepo/Active%20Stereo%20RL/logs/benchmarks/occlusion_gating_benchmark.json) | Quantifies 26.9% monocular occlusion boundary rate and proves ghosting suppression via LRC check. |
| **Middlebury Head-to-Head Suite** | [`scripts/benchmarks/benchmark_empirical_head_to_head_middlebury.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/scripts/benchmarks/benchmark_empirical_head_to_head_middlebury.py) | Direct empirical audit comparing Bicubic, DyT-Edge-INR, NAFRSSR-M, and NAFRSSR-T. |
| **Temporal Stability Suite** | [`scripts/benchmarks/benchmark_temporal_stability.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/scripts/benchmarks/benchmark_temporal_stability.py) | Evaluates frame-to-frame scintillation and temporal warping error. |
| **Vertical Parallax Audit Suite** | [`scripts/benchmarks/benchmark_epipolar_vertical_tolerance.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/scripts/benchmarks/benchmark_epipolar_vertical_tolerance.py) | Measures cross-attention tolerance under vertical epipolar distortion and camera calibration drift. |
| **Radiometric Audit Suite** | [`scripts/benchmarks/benchmark_radiometric_exposure_tolerance.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/scripts/benchmarks/benchmark_radiometric_exposure_tolerance.py) | Evaluates auto-exposure gain mismatch and photometric robustness. |
| **Occlusion Gating Audit Suite** | [`scripts/benchmarks/benchmark_occlusion_gating.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/scripts/benchmarks/benchmark_occlusion_gating.py) | Measures Left-Right Consistency (LRC) soft disparity gating on real Middlebury scenes. |
| **Occlusion-Gated SCAM Test** | [`tests/test_occlusion_gated_scam.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/tests/test_occlusion_gated_scam.py) | Verifies bidirectional disparity consistency and occlusion confidence mask generation. |
| **OG-WindowedSCAM Backbone** | [`src/edge_stereoinr/phase2_edge_inr/backbones/og_windowed_scam.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/src/edge_stereoinr/phase2_edge_inr/backbones/og_windowed_scam.py) | Production module for Left-Right Consistency occlusion-gated stereo cross-attention. |
| **Foveated Continuous INR Test** | [`tests/test_spatially_varying_foveated_inr.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/tests/test_spatially_varying_foveated_inr.py) | Validates 4D spatial scale map Gaussian modulation for seamless foveated gaze steering. |
| **Precision Tradeoff Audit** | [`tests/test_precision_fp16_vs_int8_inr.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/tests/test_precision_fp16_vs_int8_inr.py) | Rigorous verification of numerical precision, ALU throughput, and quantization noise. |
| **Continuous RepDyT Suite** | [`tests/test_rep_dyt_continuous_inr.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/tests/test_rep_dyt_continuous_inr.py) | Full verification of ContinuousRepDyTEdgeINR with 0.00e+00 post-deploy parity. |
| **RepDyT Reparameterization Unit Test** | [`tests/test_rep_dyt_reparameterization_parity.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/tests/test_rep_dyt_reparameterization_parity.py) | Mathematical parity proof verifying 0.00e+00 error after structural reparameterization. |
| **Dynamic Tanh LUT Generator** | [`deployment/vulkan/generate_dyt_lut.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/deployment/vulkan/generate_dyt_lut.py) | Pre-computes 256-entry 1D Texture LUT for Vulkan compute shaders (RMSE < 1e-4). |
| **Cross-Model Comparison Matrix** | [`logs/benchmarks/cross_model_comprehensive_matrix.json`](file:///d:/Gitrepo/Active%20Stereo%20RL/logs/benchmarks/cross_model_comprehensive_matrix.json) | Full 11-model comparison matrix (parameters, SRAM, latency, GMEM compliance) across 2016-2026 literature. |
| **Cross-Model Benchmark Suite** | [`scripts/benchmarks/benchmark_cross_model_comparison.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/scripts/benchmarks/benchmark_cross_model_comparison.py) | Automated auditor evaluating SRAM budgets, DRAM spills, and 90 FPS VR feasibility. |
| **157-Pair Benchmark Suite** | [`scripts/benchmarks/benchmark_all_157_canonical_pairs.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/scripts/benchmarks/benchmark_all_157_canonical_pairs.py) | Batched tiled streaming audit enforcing 100% GMEM residency and official MATLAB Y-channel PSNR. |
| **Numerical Parity Audit** | [`scripts/benchmarks/benchmark_onnx_runtime_parity.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/scripts/benchmarks/benchmark_onnx_runtime_parity.py) | Bitwise parity audit between PyTorch and ONNX Runtime ($\text{MSE} = 9.22 \times 10^{-16}$). |
| **Quantization Noise Audit** | [`scripts/benchmarks/benchmark_qnn_onnx_quantization.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/scripts/benchmarks/benchmark_qnn_onnx_quantization.py) | Simulated INT8 noise audit on real Middlebury tiles ($57.1\text{ dB}$ SNR). |
| **Visual Comparison Suite** | [`scripts/benchmarks/generate_visual_comparison_suite.py`](file:///d:/Gitrepo/Active%20Stereo%20RL/scripts/benchmarks/generate_visual_comparison_suite.py) | Renders multi-panel visual comparisons with high-resolution error residual heatmaps. |

---

## 5. Summary of Benchmark Truth Ledger

```
+---------------------------------------------------------------------------------------------------------------+
|                                    MASTER 157-PAIR CANONICAL BENCHMARK SUMMARY                                |
+---------------------------------------------------------------------------------------------------------------+
| Benchmark Dataset       | Pairs     | Bicubic Baseline | DyT-Edge-INR (Soup) | Delta (Gain) | Win Rate (%)   |
+---------------------------------------------------------------------------------------------------------------+
| Middlebury (Full Split) | 5 pairs   | 28.06 dB         | 28.67 dB            | +0.61 dB     | 100.0% (5/5)   |
| KITTI 2012 (Full Split) | 20 pairs  | 26.65 dB         | 27.22 dB            | +0.57 dB     | 100.0% (20/20) |
| KITTI 2015 (Full Split) | 20 pairs  | 27.11 dB         | 27.77 dB            | +0.66 dB     | 100.0% (20/20) |
| Flickr1024 (Full Split) | 112 pairs | 23.36 dB         | 23.63 dB            | +0.27 dB     |  98.2% (110/112|
+---------------------------------------------------------------------------------------------------------------+
| MASTER WEIGHTED AVERAGE | 157 pairs | 24.41 dB         | 24.78 dB            | +0.37 dB     |  98.7% (155/157|
+---------------------------------------------------------------------------------------------------------------+
```
