"""
Sub-Pixel Spatial Smoothness & Temporal Coherence Benchmark for VR Passthrough:
Tests whether ContinuousDyTEdgeINR behaves as a true continuous mathematical operator
under fractional sub-pixel camera jitter (simulating 90 FPS VR head motion).

Computes Sub-Pixel Coherence Error (SPCE) and Temporal Jitter Index across fractional shifts
dx in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0] pixels.
"""

import os, sys, glob, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT_DIR = r"d:\Gitrepo\Active Stereo RL"
sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_continuous_scale_inr import ContinuousDyTEdgeINR

def subpixel_shift(tensor, dx, dy):
    """Applies sub-pixel shift dx, dy using bilinear grid sampling"""
    B, C, H, W = tensor.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=tensor.device),
        torch.linspace(-1, 1, W, device=tensor.device),
        indexing='ij'
    )
    # Convert pixel shift to normalized coords [-1, 1]
    norm_dx = (dx * 2.0) / float(W)
    norm_dy = (dy * 2.0) / float(H)
    grid_x = grid_x - norm_dx
    grid_y = grid_y - norm_dy
    grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).repeat(B, 1, 1, 1)
    return F.grid_sample(tensor, grid, mode='bicubic', padding_mode='border', align_corners=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'dyt_continuous_inr_quest3s_soup_best.pth')

model = ContinuousDyTEdgeINR(width=48, num_blks=4).to(device).eval()
sd = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd, strict=False)

mid_dirs = sorted(glob.glob(os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Middlebury', 'hr', '*')))

shifts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

print("=" * 85)
print(">>> SUB-PIXEL TEMPORAL COHERENCE BENCHMARK (VR HEADSET MICRO-MOTION SIMULATION)")
print("=" * 85)
print(f"{'Shift dx (px)':<15} | {'Bicubic SPCE':<20} | {'Model Soup SPCE':<20} | {'Coherence Ratio':<15}")
print("-" * 85)

for dx in shifts:
    err_bic_list = []
    err_model_list = []
    
    for pdir in mid_dirs:
        hl = os.path.join(pdir, 'hr0.png')
        hr = os.path.join(pdir, 'hr1.png')
        if not os.path.exists(hl): continue
        img_l = torch.from_numpy(np.array(Image.open(hl).convert('RGB'), dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
        img_r = torch.from_numpy(np.array(Image.open(hr).convert('RGB'), dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
        
        # Center crop 192x192
        H, W = img_l.shape[2], img_l.shape[3]
        cy, cx = (H - 192) // 2, (W - 192) // 2
        crop_hl = img_l[:, :, cy:cy+192, cx:cx+192]
        crop_hr = img_r[:, :, cy:cy+192, cx:cx+192]
        
        crop_ll = F.interpolate(crop_hl, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
        crop_lr = F.interpolate(crop_hr, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
        
        # Apply fractional shift to LR input
        shifted_ll = subpixel_shift(crop_ll, dx, 0.0)
        shifted_lr = subpixel_shift(crop_lr, dx, 0.0)
        
        with torch.no_grad():
            # Model prediction on unshifted input
            pred_base_l, _ = model(crop_ll, crop_lr, scale=4.0)
            # Model prediction on shifted input
            pred_shifted_l, _ = model(shifted_ll, shifted_lr, scale=4.0)
            
            # Expected shifted output (scaled by 4x in HR space)
            expected_shifted_l = subpixel_shift(pred_base_l, dx * 4.0, 0.0)
            
            # Bicubic prediction on unshifted input
            bic_base_l = F.interpolate(crop_ll, scale_factor=4, mode='bicubic', align_corners=False)
            bic_shifted_l = F.interpolate(shifted_ll, scale_factor=4, mode='bicubic', align_corners=False)
            expected_bic_l = subpixel_shift(bic_base_l, dx * 4.0, 0.0)
            
            # Calculate Sub-Pixel Coherence Error (SPCE)
            spce_model = F.l1_loss(pred_shifted_l, expected_shifted_l).item()
            spce_bic = F.l1_loss(bic_shifted_l, expected_bic_l).item()
            
            err_model_list.append(spce_model)
            err_bic_list.append(spce_bic)
            
    m_bic = np.mean(err_bic_list)
    m_mod = np.mean(err_model_list)
    ratio = m_mod / max(m_bic, 1e-8)
    print(f"{dx:<15.1f} | {m_bic:<20.5f} | {m_mod:<20.5f} | {ratio:<15.2f}")

print("=" * 85)
