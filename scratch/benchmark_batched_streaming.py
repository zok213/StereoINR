import os, sys, time
import torch
import numpy as np

ROOT_DIR = r"d:\Gitrepo\Active Stereo RL"
sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_continuous_scale_inr import ContinuousDyTEdgeINR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Benchmarking on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

ckpt_path = os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'dyt_continuous_inr_quest3s_soup_best.pth')
model = ContinuousDyTEdgeINR(width=48, num_blks=4).to(device).eval()
sd = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd, strict=False)

# Test various batch sizes representing tile streaming
# For full frame 320x240 -> 1280x960:
# With 64x64 tiles, step 52: exactly 30 tiles (5 rows x 6 cols)
batch_sizes = [1, 5, 10, 30]

print("=" * 80)
print(f"{'Batch Size':<12} | {'Scale':<8} | {'Latency (ms)':<15} | {'Throughput (FPS)':<18} | {'Per-Tile (ms)':<15}")
print("-" * 80)

for b in batch_sizes:
    x_l = torch.randn(b, 3, 64, 64, device=device)
    x_r = torch.randn(b, 3, 64, 64, device=device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(x_l, x_r, scale=4.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    runs = 50
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(x_l, x_r, scale=4.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    avg_ms = (t1 - t0) * 1000.0 / runs
    fps = 1000.0 / avg_ms
    per_tile = avg_ms / b
    print(f"{b:<12} | {'4.0x (Int)':<8} | {avg_ms:<15.2f} | {fps:<18.1f} | {per_tile:<15.3f}")

print("-" * 80)
for b in batch_sizes:
    x_l = torch.randn(b, 3, 64, 64, device=device)
    x_r = torch.randn(b, 3, 64, 64, device=device)
    
    # Continuous scale 2.5x
    for _ in range(5):
        with torch.no_grad():
            _ = model(x_l, x_r, scale=2.5)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    runs = 50
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(x_l, x_r, scale=2.5)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    avg_ms = (t1 - t0) * 1000.0 / runs
    fps = 1000.0 / avg_ms
    per_tile = avg_ms / b
    print(f"{b:<12} | {'2.5x (Cont)':<8} | {avg_ms:<15.2f} | {fps:<18.1f} | {per_tile:<15.3f}")

print("=" * 80)
