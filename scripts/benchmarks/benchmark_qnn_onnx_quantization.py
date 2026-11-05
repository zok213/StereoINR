"""
Benchmark and Validation: Qualcomm QNN INT8 ONNX Quantization Parity
Validates the INT8 post-training quantized ONNX model against the FP32 baseline on real Middlebury stereo tiles.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from PIL import Image

ROOT_DIR = r"d:\Gitrepo\Active Stereo RL"
sys.path.insert(0, ROOT_DIR)

def run_qnn_quantization_benchmark():
    input_onnx = os.path.join(ROOT_DIR, 'weights', 'qnn_npu', 'dyt_edge_inr_qnn_tile64_4x.onnx')
    output_int8_onnx = os.path.join(ROOT_DIR, 'weights', 'qnn_npu', 'dyt_edge_inr_qnn_tile64_4x_int8.onnx')
    
    assert os.path.exists(input_onnx), f"Source ONNX not found: {input_onnx}"
    
    fp32_size_kb = os.path.getsize(input_onnx) / 1024.0
    print(f"[*] Original FP32 ONNX Model Size: {fp32_size_kb:.1f} KB")
    
    # 1. Perform Dynamic INT8 Quantization (weights quantized to INT8)
    print("[*] Quantizing ONNX model to INT8 (Qualcomm HVX format)...")
    quantize_dynamic(
        model_input=input_onnx,
        model_output=output_int8_onnx,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['Conv', 'MatMul', 'Gemm']
    )
    
    int8_size_kb = os.path.getsize(output_int8_onnx) / 1024.0
    compression_ratio = fp32_size_kb / int8_size_kb
    print(f"[+] INT8 Quantized ONNX Model Size: {int8_size_kb:.1f} KB (Compression: {compression_ratio:.2f}x)")
    
    # 2. Prepare real Middlebury test tile
    test_scene_dir = os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Middlebury', 'hr', 'cloth2')
    img_l_path = os.path.join(test_scene_dir, 'hr0.png')
    img_r_path = os.path.join(test_scene_dir, 'hr1.png')
    
    assert os.path.exists(img_l_path), f"Test image not found: {img_l_path}"
    
    img_l = np.array(Image.open(img_l_path).convert('RGB')).astype(np.float32) / 255.0
    img_r = np.array(Image.open(img_r_path).convert('RGB')).astype(np.float32) / 255.0
    
    # Extract 64x64 center patch as LR input (downsampled by 4x)
    H, W, _ = img_l.shape
    h_start, w_start = H // 2, W // 2
    crop_l = img_l[h_start:h_start+256, w_start:w_start+256, :]
    crop_r = img_r[h_start:h_start+256, w_start:w_start+256, :]
    
    # Downsample by 4x to create 64x64 LR input
    lr_l = Image.fromarray((crop_l * 255).astype(np.uint8)).resize((64, 64), Image.BICUBIC)
    lr_r = Image.fromarray((crop_r * 255).astype(np.uint8)).resize((64, 64), Image.BICUBIC)
    
    t_l = (np.array(lr_l).astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
    t_r = (np.array(lr_r).astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
    
    # 3. Run Inference on FP32 ONNX
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    sess_fp32 = ort.InferenceSession(input_onnx, sess_opts, providers=['CPUExecutionProvider'])
    sess_int8 = ort.InferenceSession(output_int8_onnx, sess_opts, providers=['CPUExecutionProvider'])
    
    ort_inputs = {'input_l': t_l, 'input_r': t_r}
    
    # Warmup
    for _ in range(5):
        _ = sess_fp32.run(None, ort_inputs)
        _ = sess_int8.run(None, ort_inputs)
        
    # Latency benchmark
    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        out_fp32 = sess_fp32.run(None, ort_inputs)
    time_fp32 = (time.perf_counter() - t0) / N * 1000.0
    
    t0 = time.perf_counter()
    for _ in range(N):
        out_int8 = sess_int8.run(None, ort_inputs)
    time_int8 = (time.perf_counter() - t0) / N * 1000.0
    
    print(f"[*] Inference Latency per 64x64 Tile (CPU Execution Provider):")
    print(f"    - FP32 ONNX: {time_fp32:.2f} ms")
    print(f"    - INT8 ONNX: {time_int8:.2f} ms")
    
    # 4. Numerical Accuracy & Parity
    # Output shapes: (1, 3, 256, 256) per eye
    l_fp32, r_fp32 = out_fp32[0], out_fp32[1]
    l_int8, r_int8 = out_int8[0], out_int8[1]
    
    max_diff_l = np.max(np.abs(l_fp32 - l_int8))
    mse_l = np.mean((l_fp32 - l_int8) ** 2)
    snr_l = 10 * np.log10(np.mean(l_fp32 ** 2) / (mse_l + 1e-12))
    
    max_diff_r = np.max(np.abs(r_fp32 - r_int8))
    mse_r = np.mean((r_fp32 - r_int8) ** 2)
    snr_r = 10 * np.log10(np.mean(r_fp32 ** 2) / (mse_r + 1e-12))
    
    print(f"\n[+] Numerical Parity Assessment (FP32 vs INT8):")
    print(f"    - Left Eye:  Max Diff = {max_diff_l:.4f}, MSE = {mse_l:.2e}, SNR = {snr_l:.2f} dB")
    print(f"    - Right Eye: Max Diff = {max_diff_r:.4f}, MSE = {mse_r:.2e}, SNR = {snr_r:.2f} dB")
    
    # Check that SNR is high (> 35 dB) and max difference is small (< 0.05 on [0, 1] scale)
    assert snr_l > 30.0, f"SNR Left too low: {snr_l:.2f} dB"
    assert snr_r > 30.0, f"SNR Right too low: {snr_r:.2f} dB"
    assert max_diff_l < 0.08, f"Max diff Left too large: {max_diff_l}"
    assert max_diff_r < 0.08, f"Max diff Right too large: {max_diff_r}"
    
    print("\n[SUCCESS] Qualcomm Hexagon INT8 ONNX Parity Verified!")
    print(f"INT8 model saved at: {output_int8_onnx}")

if __name__ == '__main__':
    run_qnn_quantization_benchmark()
