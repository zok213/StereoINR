"""
Vertical Epipolar Misalignment & Mechanical Calibration Jitter Benchmark:
In real-world VR headsets (Meta Quest 3S), physical headset flexure, thermal expansion,
and imperfect stereo rectification create small vertical disparities:
Delta_y in {-2, -1, 0, +1, +2} pixels.

This benchmark quantifies:
1. Performance degradation curve under vertical epipolar drift.
2. Resilience of DyT-Edge-INR vs Bicubic interpolation.
3. Left-eye reconstruction stability.
"""

import os, sys, math, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT_DIR = r"d:\Gitrepo\Active Stereo RL"
sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_continuous_scale_inr import ContinuousDyTEdgeINR

def calc_psnr(p, t):
    mse = F.mse_loss(p, t).item()
    return 10 * math.log10(1.0 / max(mse, 1e-10))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'dyt_continuous_inr_quest3s_soup_best.pth')
model = ContinuousDyTEdgeINR(width=48, num_blks=4).to(device).eval()
sd = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd, strict=False)

mid_dir = os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Middlebury')
mid_scenes = sorted(os.listdir(os.path.join(mid_dir, 'hr')))

vertical_shifts = [-2, -1, 0, 1, 2]

print("=" * 115)
print(">>> VERTICAL EPIPOLAR MISALIGNMENT BENCHMARK (META QUEST 3S MECHANICAL JITTER)")
print("=" * 115)
print(f"{'Vertical Shift (dy)':<24} | {'Bicubic PSNR':<15} | {'DyT-Edge-INR PSNR':<18} | {'Delta vs Bicubic':<18} | {'Left-Eye PSNR':<15}")
print("-" * 115)

for dy in vertical_shifts:
    bic_psnr_list = []
    dyt_psnr_list = []
    left_psnr_list = []
    
    for scene in mid_scenes:
        hl_path = os.path.join(mid_dir, 'hr', scene, 'hr0.png')
        hr_path = os.path.join(mid_dir, 'hr', scene, 'hr1.png')
        
        img_l_raw = torch.from_numpy(np.array(Image.open(hl_path).convert('RGB'), dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
        img_r_raw = torch.from_numpy(np.array(Image.open(hr_path).convert('RGB'), dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
        
        # Apply vertical shift to Right eye in HR (dy * 4 in HR, so dy in LR)
        # Shift Right eye by dy in LR
        lr_l = F.interpolate(img_l_raw, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
        lr_r_orig = F.interpolate(img_r_raw, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
        
        if dy > 0:
            lr_r = F.pad(lr_r_orig[:, :, dy:, :], (0, 0, 0, dy), mode='replicate')
        elif dy < 0:
            lr_r = F.pad(lr_r_orig[:, :, :dy, :], (0, 0, -dy, 0), mode='replicate')
        else:
            lr_r = lr_r_orig
            
        # Bicubic
        bic_l = F.interpolate(lr_l, size=img_l_raw.shape[2:], mode='bicubic', align_corners=False).clamp(0, 1)
        bic_r = F.interpolate(lr_r, size=img_r_raw.shape[2:], mode='bicubic', align_corners=False).clamp(0, 1)
        
        # Target for right eye is shifted accordingly
        if dy > 0:
            hr_r_target = F.pad(img_r_raw[:, :, dy*4:, :], (0, 0, 0, dy*4), mode='replicate')
        elif dy < 0:
            hr_r_target = F.pad(img_r_raw[:, :, :dy*4, :], (0, 0, -dy*4, 0), mode='replicate')
        else:
            hr_r_target = img_r_raw
            
        with torch.no_grad():
            sr_l, sr_r = model(lr_l, lr_r, scale=4.0)
            
        p_bic = 0.5 * (calc_psnr(bic_l, img_l_raw) + calc_psnr(bic_r, hr_r_target))
        p_dyt = 0.5 * (calc_psnr(sr_l, img_l_raw) + calc_psnr(sr_r, hr_r_target))
        p_left = calc_psnr(sr_l, img_l_raw)
        
        bic_psnr_list.append(p_bic)
        dyt_psnr_list.append(p_dyt)
        left_psnr_list.append(p_left)
        
    m_bic = np.mean(bic_psnr_list)
    m_dyt = np.mean(dyt_psnr_list)
    m_left = np.mean(left_psnr_list)
    diff = m_dyt - m_bic
    
    print(f"dy = {dy:+2d} px LR ({dy*4:+2d} px HR)  | {m_bic:<15.2f} | {m_dyt:<18.2f} | {diff:+<17.2f} dB | {m_left:<15.2f}")

print("=" * 115)
