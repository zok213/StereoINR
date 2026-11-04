# Unity 6 Sentis Deployment Guide: EdgeStereoINR on Meta Quest 3S

## Architecture Overview
The **EdgeStereoINR** engine provides real-time arbitrary-scale continuous super-resolution ($1.2\times - 4.0\times$) for the dual RGB passthrough cameras on **Meta Quest 3S** (Qualcomm Snapdragon XR2 Gen 2 / Adreno 740 GPU).

### Key Technical Specs
- **Model Parameters:** 104,435 parameters ($204.0\text{ KB}$ FP16, $102.0\text{ KB}$ INT8).
- **Target Hardware:** Qualcomm Adreno 740 (1024 KB GMEM on-chip SRAM).
- **Tile Configuration:** $64 \times 64$ LR tiles with $12\text{px}$ overlap ($768.0\text{ KB}$ peak tile SRAM, 100% GMEM SRAM resident, 0 DRAM spills).
- **Stitching Quality:** 2D Hann-window feathering achieving $>57.5\text{ dB}$ SNR vs non-tiled direct inference (0 seam artifacts).
- **Inference Runtime:** `Unity.Sentis` with `BackendType.GPUCompute` (Vulkan compute shaders).

---

## Files in this Directory
1. `TiledStereoSentisStreamer.cs`: Unity MonoBehaviour managing stereo camera feeds, tile dispatch, Sentis worker scheduling, and GPU compositing.
2. `HannBlendShader.shader`: GPU additive-blend shader applying 2D Hann window feathering directly on target eye render textures with zero CPU readback.
3. `dyt_continuous_inr_quest3s_soup_fp16.onnx`: ONNX model weights (Model Soup $\theta_{soup} = 0.75 \theta_{v8} + 0.25 \theta_{v9}$) located in `weights/kaggle_dual_t4/`.

---

## Step-by-Step Integration in Unity 6

### 1. Package Installation
Open `Window > Package Manager` and install:
- **Sentis** (`com.unity.sentis`) version 2.1.0 or newer.
- **Oculus XR Plugin** (`com.unity.xr.oculus`) with Quest 3S support.

### 2. Import Model Asset
- Drag `weights/kaggle_dual_t4/dyt_continuous_inr_quest3s_soup_fp16.onnx` into your Unity `Assets/Models/` folder.
- Sentis will automatically import it as a `ModelAsset`.

### 3. Setup Scene & Cameras
1. Create an empty GameObject `StereoSRStreamer` and attach `TiledStereoSentisStreamer.cs`.
2. Assign:
   - `Model Asset`: `dyt_continuous_inr_quest3s_soup_fp16`
   - `Backend Type`: `GPUCompute`
   - `Tile Size`: 64
   - `Overlap`: 12
   - `Scale Factor`: 4.0 (or dynamic continuous float between 1.2 and 4.0)
   - `Hann Blend Material`: Create a material with shader `EdgeStereoINR/HannBlendShader`.
   - `Left/Right Camera Feed`: WebCamTexture or Passthrough RenderTextures ($320\times 240$ or $640\times 480$).
   - `Left/Right Output Target`: Eye display RenderTextures ($1280\times 960$).

### 4. Zero-Readback Verification
To verify that no CPU stalls occur during Passthrough:
- In Unity Profiler, ensure `AsyncGPUReadback` and `GetPixels` show 0 ms.
- Profiler GPU frame time for Sentis worker should remain within the 11.1 ms (90 FPS) or 13.8 ms (72 FPS) VR budget.
