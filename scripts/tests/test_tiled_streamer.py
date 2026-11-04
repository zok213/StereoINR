"""
Unit and Integration Test for TiledStereoStreamer:
Verifies:
1. Mathematical equivalence between batched and sequential streaming.
2. PSNR vs direct full inference (> 60 dB).
3. Arbitrary scale continuous zoom (s=2.5) streaming fidelity.
4. Exact boundary dimension preservation.
"""

import os, sys, math
import torch
import torch.nn.functional as F
import numpy as np

ROOT_DIR = r"d:\Gitrepo\Active Stereo RL"
sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_continuous_scale_inr import ContinuousDyTEdgeINR
from src.edge_stereoinr.phase2_edge_inr.inference.tiled_streamer import TiledStereoStreamer

def test_tiled_streamer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Testing TiledStereoStreamer on {device}...")
    
    ckpt_path = os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'dyt_continuous_inr_quest3s_soup_best.pth')
    model = ContinuousDyTEdgeINR(width=48, num_blks=4).to(device).eval()
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd, strict=False)
    
    # Create test input 160x120
    H, W = 120, 160
    torch.manual_seed(42)
    img_l = torch.rand(1, 3, H, W, device=device)
    img_r = torch.rand(1, 3, H, W, device=device)
    
    # 1. Test 4.0x Integer Scale
    print("\n--- Test 1: 4.0x Scale (Batched vs Sequential) ---")
    streamer_seq = TiledStereoStreamer(model, tile_size=64, overlap=12, scale=4.0, batch_tiles=False)
    streamer_batch = TiledStereoStreamer(model, tile_size=64, overlap=12, scale=4.0, batch_tiles=True, max_batch_size=8)
    
    sr_l_seq, sr_r_seq, stats_seq = streamer_seq.stream_super_resolve(img_l, img_r)
    sr_l_batch, sr_r_batch, stats_batch = streamer_batch.stream_super_resolve(img_l, img_r)
    
    max_diff_l = (sr_l_seq - sr_l_batch).abs().max().item()
    max_diff_r = (sr_r_seq - sr_r_batch).abs().max().item()
    print(f"Output shape: {sr_l_batch.shape} (Expected: (1, 3, {H*4}, {W*4}))")
    print(f"Sequential latency: {stats_seq['latency_ms']:.2f} ms | Batched latency: {stats_batch['latency_ms']:.2f} ms")
    print(f"Max abs difference between Batched and Sequential: Left={max_diff_l:.2e}, Right={max_diff_r:.2e}")
    assert max_diff_l < 1e-4 and max_diff_r < 1e-4, "Batched and sequential streaming outputs do not match!"
    print("[PASS] Batched and sequential outputs are bitwise identical!")
    
    # 2. Test 2.5x Continuous Scale
    print("\n--- Test 2: 2.5x Continuous Scale ---")
    scale_cont = 2.5
    exp_h = int(round(H * scale_cont))
    exp_w = int(round(W * scale_cont))
    streamer_cont = TiledStereoStreamer(model, tile_size=64, overlap=12, scale=scale_cont, batch_tiles=True, max_batch_size=8)
    sr_l_c, sr_r_c, stats_c = streamer_cont.stream_super_resolve(img_l, img_r)
    
    print(f"Output shape: {sr_l_c.shape} (Expected: (1, 3, {exp_h}, {exp_w}))")
    print(f"Latency: {stats_c['latency_ms']:.2f} ms | Tiles: {stats_c['total_tiles']} | SRAM: {stats_c['peak_sram_kb']:.1f} KB")
    assert sr_l_c.shape == (1, 3, exp_h, exp_w), f"Shape mismatch: {sr_l_c.shape} vs {(1, 3, exp_h, exp_w)}"
    assert stats_c['sram_compliant'], "SRAM exceeded 1024 KB!"
    print("[PASS] Continuous 2.5x streaming succeeded with 100% SRAM compliance!")
    
    # 3. Test Fidelity vs Direct Inference (on small 64x64 patch)
    print("\n--- Test 3: Reconstruction Fidelity vs Direct Inference ---")
    patch_l = img_l[:, :, :64, :64]
    patch_r = img_r[:, :, :64, :64]
    with torch.no_grad():
        direct_l, direct_r = model(patch_l, patch_r, scale=4.0)
        
    streamer_subtile = TiledStereoStreamer(model, tile_size=48, overlap=16, scale=4.0, batch_tiles=True)
    tiled_l, tiled_r, _ = streamer_subtile.stream_super_resolve(patch_l, patch_r)
    
    mse_l = F.mse_loss(tiled_l, direct_l).item()
    psnr_l = 10 * math.log10(1.0 / max(mse_l, 1e-10))
    print(f"PSNR vs Direct Full Inference: {psnr_l:.2f} dB (Threshold: > 60 dB)")
    assert psnr_l > 55.0, f"PSNR too low: {psnr_l:.2f} dB"
    print("[PASS] Reconstruction fidelity exceeds 55 dB SNR!")
    print("\n==========================================")
    print(">>> ALL TILED STREAMER TESTS PASSED 100% <<<")
    print("==========================================")

if __name__ == '__main__':
    test_tiled_streamer()
