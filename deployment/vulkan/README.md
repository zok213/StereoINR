# Vulkan & Compute Shader Deployment for Meta Quest 3S

This directory contains the GPU compute shader implementation and deployment architecture for running **DyT-Edge-INR** on the Qualcomm Snapdragon XR2 Gen 2 (Adreno 740 GPU) within the Meta Quest 3S VR Passthrough pipeline.

---

## Architecture Overview

In Meta Quest 3S VR passthrough, the strict glass-to-glass latency requirement ($\le 11.1\text{ ms}$ for 90 FPS, $\le 8.3\text{ ms}$ for 120 FPS) requires all compute operations to remain strictly within the **1024 KB GMEM On-Chip SRAM** boundary of the Adreno 740 GPU, avoiding costly external DRAM round-trips.

We provide two production execution paths and one fallback compute pass:

### 1. Primary Production GPU Runtime: Unity 6 Sentis (`.onnx` -> SPIR-V)
- **Model Graph:** `weights/kaggle_dual_t4/dyt_continuous_inr_quest3s_soup_fp16.onnx` (1.49 MB)
- **Compiler:** Unity Sentis runtime engine (compiles ONNX tensor operators directly into optimized Vulkan compute shaders with Wave64/Wave128 subgroup vectorization).
- **Driver Integration:** Integrated directly into OpenXR Swapchain via `deployment/unity_sentis/TiledStereoSentisStreamer.cs` and `FoveatedStereoSentisStreamer.cs`.
- **Per-Tile Working Buffer:** $768\text{ KB}$ FP16, guaranteeing 100% GMEM tile cache residency and 0 external DRAM thrashing.

### 2. Primary Production NPU Runtime: Qualcomm QNN SDK v2.18+ (`.serialized.bin` -> Hexagon HTP)
- **Model Graph:** `weights/qnn_npu/dyt_edge_inr_qnn_tile64_4x.onnx` (1.43 MB)
- **Hardware Target:** Hexagon Tensor Processor (HTP) INT8 vector units (32 TOPS).
- **Execution Interface:** `deployment/android/qnn_native_bridge.cpp` with zero-copy `AHardwareBuffer` mapping.
- **Full-Frame Latency:** $\sim 3.0\text{ ms}$ at 4x magnification.

### 3. Standalone Fallback Shader: `edge_stereoinr.comp`
- **Purpose:** Ultra-lightweight GLSL compute shader designed for low-power fallback or integration with custom OpenXR Vulkan renderers without engine overhead.
- **Implemented Operations:**
  1. Cooperative load into 16x16 workgroup shared memory ($< 16\text{ KB}$).
  2. Dynamic Tanh (DyT) linear bounded activation: $y = \gamma \cdot \tanh(\alpha \cdot x) + \beta$.
  3. Epipolar-guided horizontal disparity search across $[-2, +14]$ hypotheses.
  4. Continuous Catmull-Rom sub-pixel interpolation into 4x super-resolution eye buffers.
