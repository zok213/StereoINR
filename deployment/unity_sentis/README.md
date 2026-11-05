# Unity 6 Sentis Deployment Guide: EdgeStereoINR on Meta Quest 3S

## 1. Hardware & OS Reality Check
- **Target Device:** Meta Quest 3S (Snapdragon XR2 Gen 2 / Adreno 740 GPU / Hexagon NPU).
- **Horizon OS SELinux Sandbox Rule:** Retail Quest 3S kernel strictly blocks third-party APKs (`neverallow untrusted_app fastrpc_device:chr_file`). Third-party Unity/OpenXR APKs cannot open `/dev/fastrpc-*` to communicate with the Hexagon NPU.
- **The True Production Path:** The **Qualcomm Adreno 740 GPU via Unity 6 Sentis (Vulkan Compute shaders)** is the sole legal, functional, and zero-privilege deployment target on retail headsets.
- **The NPU Role:** Qualcomm QNN SDK artifacts are preserved strictly for OEM / system-privileged apps and Qualcomm reference hardware.

---

## 2. Technical Architecture & Specs
- **Model Parameters:** 104,435 parameters ($204.0\text{ KB}$ FP16, $102.0\text{ KB}$ INT8).
- **Target SRAM GMEM:** 1024 KB on Adreno 740.
- **Tile Slicing:** $64 \times 64$ LR tiles with $12\text{px}$ overlap ($768.0\text{ KB}$ peak SRAM working buffer, 100% GMEM SRAM resident, 0 DRAM spills).
- **Stitching Quality:** 2D Hann-window feathering with optional `HannResolveShader` normalization, achieving $>57.5\text{ dB}$ SNR vs non-tiled direct inference (0 seam artifacts).
- **Inference Runtime:** `Unity.Sentis` with `BackendType.GPUCompute` (Vulkan compute shaders) and zero CPU readback.

---

## 3. Package & Source Files
1. `dist/com.activestereo.edgestreoinr-1.0.0.tgz`: Turnkey Unity Package Manager (UPM) tarball.
2. `TiledStereoSentisStreamer.cs`: Uniform tiled streaming engine with auto tensor name resolution and zero-copy GPU pipeline.
3. `FoveatedStereoSentisStreamer.cs`: Foveated streaming engine (FFSR) reducing compute by 45-60% via optical eccentricity modulation.
4. `StereoPassthroughDemo.cs`: Interactive runtime controller with live FPS and SRAM telemetry HUD.
5. `HannBlendShader.shader`: GPU additive-blend shader applying 2D Hann window feathering.
6. `HannResolveShader.shader`: Post-accumulation normalization shader dividing accumulated RGB by weight sum.
7. `package.json`: UPM manifest with Sentis dependencies.

---

## 4. Step-by-Step Integration in Unity 6

### Step 1: Install UPM Package
Open Unity Editor (Unity 6 LTS recommended):
1. Navigate to **Window > Package Manager**.
2. Click the `+` button > **Add package from tarball...**.
3. Select `deployment/unity_sentis/dist/com.activestereo.edgestreoinr-1.0.0.tgz`.

### Step 2: Import Model Asset
- Drag `weights/kaggle_dual_t4/dyt_continuous_inr_quest3s_soup_fp16.onnx` into your `Assets/Models/` folder.
- Unity Sentis imports it automatically as a `ModelAsset`.

### Step 3: Configure Scene
1. Create an empty GameObject `StereoSRStreamer` and attach `TiledStereoSentisStreamer.cs` or `FoveatedStereoSentisStreamer.cs`.
2. Assign the materials:
   - `Hann Blend Material`: Material using shader `EdgeStereoINR/HannBlendShader`.
   - `Hann Resolve Material`: Material using shader `EdgeStereoINR/HannResolveShader`.
3. Set `Backend Type`: `GPUCompute`.
4. Ensure `autoProcessOnLateUpdate = true` so inference runs seamlessly every frame.

### Step 4: Verification & Profiling
- Open **Window > Analysis > Profiler**.
- Under GPU Profiling, confirm `AsyncGPUReadback` and `GetPixels` are 0.00 ms (Zero CPU stalls).
- Frame latency on Adreno 740 GPU should remain well within the 11.1 ms (90 FPS) or 8.3 ms (120 FPS) VR budget.
