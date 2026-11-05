"""
Extreme Continuous Magnification Benchmark:
Tests the continuous arbitrary-scale capacity of DyT-Edge-INR
across an extended zoom range: s in [1.5, 8.0].

Evaluates:
- Reconstruction fidelity (PSNR vs Bicubic interpolation)
- High-frequency edge preservation
- Inference stability across extreme continuous scales
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

# Test scales: standard range [1.5, 4.0] and extreme range [5.0, 8.0]
scales = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]

print("=" * 110)
print(">>> EXTREME CONTINUOUS MAGNIFICATION BENCHMARK (s in [1.5, 8.0])")
print("=" * 110)
print(f"{'Scale Factor':<15} | {'Mode':<22} | {'Bicubic PSNR':<15} | {'DyT-Edge-INR PSNR':<18} | {'Delta vs Bicubic':<18}")
print("-" * 110)

for s in scales:
    bic_psnr_list = []
    dyt_psnr_list = []
    
    for scene in mid_scenes:
        hl_path = os.path.join(mid_dir, 'hr', scene, 'hr0.png')
        hr_path = os.path.join(mid_dir, 'hr', scene, 'hr1.png')
        
        img_l_raw = torch.from_numpy(np.array(Image.open(hl_path).convert('RGB'), dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
        img_r_raw = torch.from_numpy(np.array(Image.open(hr_path).convert('RGB'), dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
        H, W = img_l_raw.shape[2], img_l_raw.shape[3]
        
        in_h = int(H // s)
        in_w = int(W // s)
        
        # Exact HR target matching the model's output dimensions
        if abs(s - 4.0) < 1e-3:
            target_h = in_h * 4
            target_w = in_w * 4
        else:
            target_h = int(round(in_h * s))
            target_w = int(round(in_w * s))
            
        crop_hl = img_l_raw[:, :, :target_h, :target_w]
        crop_hr = img_r_raw[:, :, :target_h, :target_w]
        
        lr_l = F.interpolate(crop_hl, size=(in_h, in_w), mode='bicubic', align_corners=False).clamp(0, 1)
        lr_r = F.interpolate(crop_hr, size=(in_h, in_w), mode='bicubic', align_corners=False).clamp(0, 1)
        
        bic_l = F.interpolate(lr_l, size=(target_h, target_w), mode='bicubic', align_corners=False).clamp(0, 1)
        bic_r = F.interpolate(lr_r, size=(target_h, target_w), mode='bicubic', align_corners=False).clamp(0, 1)
        
        with torch.no_grad():
            sr_l, sr_r = model(lr_l, lr_r, scale=float(s))
            
        p_bic = 0.5 * (calc_psnr(bic_l, crop_hl) + calc_psnr(bic_r, crop_hr))
        p_dyt = 0.5 * (calc_psnr(sr_l, crop_hl) + calc_psnr(sr_r, crop_hr))
        
        bic_psnr_list.append(p_bic)
        dyt_psnr_list.append(p_dyt)
        
    m_bic = np.mean(bic_psnr_list)
    m_dyt = np.mean(dyt_psnr_list)
    diff = m_dyt - m_bic
    mode = "Fast PixelShuffle" if abs(s - 4.0) < 1e-3 else "Continuous TMU INR"
    
    print(f"{s:<15.1f} | {mode:<22} | {m_bic:<15.2f} | {m_dyt:<18.2f} | {diff:+<17.2f} dB")

print("=" * 110)
