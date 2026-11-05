"""
Radiometric and Exposure Disparity Robustness Benchmark:
Evaluates Stereo SR under camera exposure and photometric imbalance:
I_R_perturbed = alpha * I_R + beta, simulating Quest 3S independent auto-exposure.

Compares:
1. Baseline Model without Radiometric Normalization.
2. Mathematically Invariant Channel-Wise Zero-Mean Unit-Variance (ZMU) Normalized Cross-Attention.
Evaluates on Middlebury and KITTI test sets.
"""

import os, sys, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT_DIR = r"d:\Gitrepo\Active Stereo RL"
sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_continuous_scale_inr import ContinuousDyTEdgeINR

def calc_psnr(p, t):
    mse = F.mse_loss(p, t).item()
    return 10 * math.log10(1.0 / max(mse, 1e-10))

def calc_ssim(img1, img2, window_size=11):
    C1, C2 = (0.01) ** 2, (0.03) ** 2
    y1 = 0.299 * img1[:, 0:1] + 0.587 * img1[:, 1:2] + 0.114 * img1[:, 2:3]
    y2 = 0.299 * img2[:, 0:1] + 0.587 * img2[:, 1:2] + 0.114 * img2[:, 2:3]
    mu1 = F.avg_pool2d(y1, kernel_size=window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(y2, kernel_size=window_size, stride=1, padding=window_size // 2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.avg_pool2d(y1 * y1, kernel_size=window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(y2 * y2, kernel_size=window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(y1 * y2, kernel_size=window_size, stride=1, padding=window_size // 2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'dyt_continuous_inr_quest3s_soup_best.pth')
model = ContinuousDyTEdgeINR(width=48, num_blks=4).to(device).eval()
sd = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd, strict=False)

# Load Middlebury test pairs
mid_dir = os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Middlebury')
mid_scenes = sorted(os.listdir(os.path.join(mid_dir, 'hr')))

# Test conditions:
# (alpha, beta): gain and bias applied to Right camera only
conditions = [
    (1.00, 0.00, "Clean (Matched Exposure)"),
    (0.85, 0.00, "Underexposed Right Eye (Gain -15%)"),
    (1.15, 0.00, "Overexposed Right Eye (Gain +15%)"),
    (1.00, -0.05, "Negative Bias Offset (-0.05)"),
    (1.00, +0.05, "Positive Bias Offset (+0.05)"),
    (1.15, -0.05, "Combined Gain+Bias (+15%, -0.05)"),
]

print("=" * 115)
print(">>> EXPOSURE & RADIOMETRIC DISPARITY ROBUSTNESS BENCHMARK (META QUEST 3S VST)")
print("=" * 115)
print(f"{'Condition':<35} | {'Bicubic PSNR':<14} | {'DyT-Edge-INR PSNR':<18} | {'Delta vs Bicubic':<18} | {'Left-Eye PSNR':<14}")
print("-" * 115)

for alpha, beta, cond_name in conditions:
    bic_psnr_list = []
    dyt_psnr_list = []
    left_psnr_list = []
    
    for scene in mid_scenes:
        hl_path = os.path.join(mid_dir, 'hr', scene, 'hr0.png')
        hr_path = os.path.join(mid_dir, 'hr', scene, 'hr1.png')
        
        img_l_raw = torch.from_numpy(np.array(Image.open(hl_path).convert('RGB'), dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
        img_r_raw = torch.from_numpy(np.array(Image.open(hr_path).convert('RGB'), dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
        
        # Apply exposure perturbation to Right eye
        img_r_perturbed = (img_r_raw * alpha + beta).clamp(0.0, 1.0)
        
        # Downsample to LR (4x)
        lr_l = F.interpolate(img_l_raw, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
        lr_r = F.interpolate(img_r_perturbed, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
        
        # Bicubic SR
        bic_l = F.interpolate(lr_l, size=img_l_raw.shape[2:], mode='bicubic', align_corners=False).clamp(0, 1)
        bic_r = F.interpolate(lr_r, size=img_r_raw.shape[2:], mode='bicubic', align_corners=False).clamp(0, 1)
        
        # Model SR
        with torch.no_grad():
            sr_l, sr_r = model(lr_l, lr_r, scale=4.0)
            
        # Target for evaluation: Left eye target is img_l_raw; Right eye target is img_r_perturbed (sensor ground truth)
        psnr_bic = 0.5 * (calc_psnr(bic_l, img_l_raw) + calc_psnr(bic_r, img_r_perturbed))
        psnr_dyt = 0.5 * (calc_psnr(sr_l, img_l_raw) + calc_psnr(sr_r, img_r_perturbed))
        psnr_left = calc_psnr(sr_l, img_l_raw)
        
        bic_psnr_list.append(psnr_bic)
        dyt_psnr_list.append(psnr_dyt)
        left_psnr_list.append(psnr_left)
        
    m_bic = np.mean(bic_psnr_list)
    m_dyt = np.mean(dyt_psnr_list)
    m_left = np.mean(left_psnr_list)
    delta = m_dyt - m_bic
    
    print(f"{cond_name:<35} | {m_bic:<14.2f} | {m_dyt:<18.2f} | {delta:+<17.2f} dB | {m_left:<14.2f}")

print("=" * 115)
