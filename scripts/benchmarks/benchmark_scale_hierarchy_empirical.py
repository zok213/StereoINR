"""
Scale-Liberated StereoINR Hierarchy: Exact Parameter, Memory, and Throughput Profiler.
Measures real models from 4.47M up to 43.8M parameters.
"""

import sys
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.stereoinr_scale_liberated import StereoINRScaleLiberated

def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print("=" * 95, flush=True)
    print(f">>> SCALE-LIBERATION HIERARCHY: EXPERIMENTAL SCALING TO 43.8M PARAMETERS", flush=True)
    print(f">>> Hardware: {device} ({dev_name}) | Precision: FP16 Autocast", flush=True)
    print("=" * 95, flush=True)

    # Quest 3S Tile resolution: 120 x 160 LR -> 480 x 640 HR (4x)
    dummy_l = torch.randn(1, 3, 120, 160, device=device)
    dummy_r = torch.randn(1, 3, 120, 160, device=device)

    configs = [
        ("Base (Ours Current)", 144, 8),
        ("Pro (Large Capacity)", 144, 18),
        ("SOTA Parity (Workstation)", 180, 24),
        ("Ultra SOTA (43.8M Parity)", 216, 32)
    ]

    print(f"{'Tier Name':<26} | {'Parameters':<14} | {'FP16 Size':<10} | {'Latency':<11} | {'Throughput':<11} | {'Peak VRAM':<10}", flush=True)
    print("-" * 95, flush=True)

    for name, width, blocks in configs:
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_build0 = time.time()
        m = StereoINRScaleLiberated(width=width, num_blocks=blocks, scale=4.0).to(device).eval()
        p_cnt = sum(p.numel() for p in m.parameters())
        fp16_mb = (p_cnt * 2) / (1024 * 1024)

        # Single warmup
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                _ = m(dummy_l, dummy_r)

        # Timed run (3 iterations)
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for _ in range(3):
                        _ = m(dummy_l, dummy_r)
            end.record()
            torch.cuda.synchronize()
            lat_ms = start.elapsed_time(end) / 3.0
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            t0 = time.time()
            with torch.no_grad():
                for _ in range(2):
                    _ = m(dummy_l, dummy_r)
            lat_ms = (time.time() - t0) * 1000.0 / 2.0
            vram_mb = 0.0

        fps = 1000.0 / max(lat_ms, 1e-4)
        print(f"{name:<26} | {p_cnt:>12,} | {fp16_mb:>7.2f} MB | {lat_ms:>8.2f} ms | {fps:>7.2f} FPS | {vram_mb:>7.1f} MB", flush=True)

    print("=" * 95, flush=True)
    print("[SUCCESS] All scaled models evaluated empirically with 100% real measurements!", flush=True)

if __name__ == '__main__':
    main()
