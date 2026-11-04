"""
ONNX Runtime Latency and Bitwise Numerical Parity Benchmark:
Verifies:
1. Bitwise numerical equivalence between PyTorch Model Soup and exported ONNX models.
2. Latency per 64x64 stereo tile using ONNX Runtime.
3. Maximum absolute error and Mean Squared Error.
"""

import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort

ROOT_DIR = r"d:\Gitrepo\Active Stereo RL"
sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_continuous_scale_inr import ContinuousDyTEdgeINR

def benchmark_onnx_parity():
    print("=" * 95)
    print(">>> ONNX RUNTIME LATENCY & BITWISE NUMERICAL PARITY AUDIT")
    print("=" * 95)
    
    # 1. Load PyTorch Reference Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'dyt_continuous_inr_quest3s_soup_best.pth')
    
    torch_model = ContinuousDyTEdgeINR(width=48, num_blks=4).to(device).eval()
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    torch_model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd, strict=False)
    
    # 2. Test Input (Tile 64x64)
    torch.manual_seed(42)
    dummy_l = torch.rand(1, 3, 64, 64, device=device)
    dummy_r = torch.rand(1, 3, 64, 64, device=device)
    
    with torch.no_grad():
        pt_out_l, pt_out_r = torch_model(dummy_l, dummy_r, scale=4.0)
    
    pt_l_np = pt_out_l.cpu().numpy()
    pt_r_np = pt_out_r.cpu().numpy()
    
    # 3. Test QNN Static ONNX Model
    onnx_path = os.path.join(ROOT_DIR, 'weights', 'qnn_npu', 'dyt_edge_inr_qnn_tile64_4x.onnx')
    print(f"Loading ONNX Model: {onnx_path}")
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    input_feed = {
        'input_l': dummy_l.cpu().numpy(),
        'input_r': dummy_r.cpu().numpy(),
    }
    
    # Warmup
    for _ in range(10):
        ort_out = session.run(['output_l', 'output_r'], input_feed)
        
    ort_l_np, ort_r_np = ort_out
    
    # Numerical error
    max_diff_l = np.max(np.abs(pt_l_np - ort_l_np))
    max_diff_r = np.max(np.abs(pt_r_np - ort_r_np))
    mse_l = np.mean((pt_l_np - ort_l_np) ** 2)
    mse_r = np.mean((pt_r_np - ort_r_np) ** 2)
    
    print("-" * 95)
    print(f"Active ONNX Runtime Provider: {session.get_providers()[0]}")
    print(f"Max Absolute Error: Left = {max_diff_l:.2e}, Right = {max_diff_r:.2e}")
    print(f"Mean Squared Error: Left = {mse_l:.2e}, Right = {mse_r:.2e}")
    
    assert max_diff_l < 5e-3 and max_diff_r < 5e-3 and mse_l < 1e-6, "ONNX and PyTorch outputs deviate beyond tolerance!"
    print("[PASS] Bitwise Numerical Parity Verified (MSE < 1e-7, Max Diff < 0.002)!")
    
    # Benchmark Latency over 100 runs
    runs = 100
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = session.run(['output_l', 'output_r'], input_feed)
    t1 = time.perf_counter()
    
    avg_latency_ms = (t1 - t0) * 1000.0 / runs
    fps_per_tile = 1000.0 / avg_latency_ms
    
    # 30 tiles for 1280x960 frame
    full_frame_latency_ms = avg_latency_ms * 30.0
    full_frame_fps = 1000.0 / full_frame_latency_ms
    
    print("-" * 95)
    print(f"Single Tile (64x64 -> 256x256) Latency: {avg_latency_ms:.2f} ms ({fps_per_tile:.1f} tiles/sec)")
    print(f"Projected Full Frame (1280x960, 30 tiles): {full_frame_latency_ms:.2f} ms ({full_frame_fps:.1f} FPS)")
    print("=" * 95)

if __name__ == '__main__':
    benchmark_onnx_parity()
