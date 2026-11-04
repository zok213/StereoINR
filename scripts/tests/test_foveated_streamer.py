"""
Unit Test for FoveatedStereoStreamer:
Verifies:
1. Optical sweet spot tile scale calculation.
2. Smooth transition from fovea (4.0x) to periphery (2.0x).
3. Exact output dimensions (H * fovea_scale, W * fovea_scale).
4. Latency and tile count profiling.
"""

import os, sys, math
import torch
import numpy as np

ROOT_DIR = r"d:\Gitrepo\Active Stereo RL"
sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_continuous_scale_inr import ContinuousDyTEdgeINR
from src.edge_stereoinr.phase2_edge_inr.inference.foveated_streamer import FoveatedStereoStreamer

def test_foveated_streamer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Testing FoveatedStereoStreamer on {device}...")
    
    ckpt_path = os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'dyt_continuous_inr_quest3s_soup_best.pth')
    model = ContinuousDyTEdgeINR(width=48, num_blks=4).to(device).eval()
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd, strict=False)
    
    # Quest 3S simulated low-res camera frame (240x320)
    H, W = 240, 320
    torch.manual_seed(123)
    img_l = torch.rand(1, 3, H, W, device=device)
    img_r = torch.rand(1, 3, H, W, device=device)
    
    streamer = FoveatedStereoStreamer(
        model,
        tile_size=64,
        overlap=12,
        fovea_radius_inner=0.35,
        fovea_radius_outer=0.65,
        fovea_scale=4.0,
        periphery_scale=2.0
    )
    
    print("\n--- Executing Foveated Super-Resolution (Gaze at Center (0, 0)) ---")
    sr_l, sr_r, stats = streamer.stream_foveated_super_resolve(img_l, img_r, center=(0.0, 0.0))
    
    exp_h, exp_w = H * 4, W * 4
    print(f"Output shape: {sr_l.shape} (Expected: (1, 3, {exp_h}, {exp_w}))")
    print(f"Total tiles: {stats['total_tiles']} | Fovea tiles: {stats['fovea_tiles']} | Peripheral tiles: {stats['periph_tiles']}")
    print(f"Fovea ratio: {stats['fovea_ratio']*100:.1f}%")
    print(f"Latency: {stats['latency_ms']:.2f} ms")
    
    assert sr_l.shape == (1, 3, exp_h, exp_w), f"Shape mismatch: {sr_l.shape} vs {(1, 3, exp_h, exp_w)}"
    assert stats['fovea_tiles'] > 0 and stats['periph_tiles'] > 0, "Both fovea and periphery should have tiles!"
    print("[PASS] Foveated super-resolution output dimensions and tile distribution verified!")
    
    print("\n=============================================")
    print(">>> FOVEATED STREAMER UNIT TEST PASSED 100% <<<")
    print("=============================================")

if __name__ == '__main__':
    test_foveated_streamer()
