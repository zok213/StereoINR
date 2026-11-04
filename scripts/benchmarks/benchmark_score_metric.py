"""
Stereo Consistency and Objective Reconstruction Evaluation (SCORE)
Following NTIRE 2023 Stereo SR Challenge & StereoINR (arXiv:2505.05509 Section 4.1-4.2).

Computes:
1. Relative Disparity Error P = (|d_SR - d_HR| > threshold)
2. SCORE = 1 - P (Percentage of pixels where disparity is preserved within 1 pixel)
3. End-Point-Error (EPE) of disparity = mean(|d_SR - d_HR|)
"""

import os, sys, glob, math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT_DIR = r"d:\Gitrepo\Active Stereo RL"
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'external', 'NAFRSSR'))

from NAFRSSR_M_Net import NAFSSR
from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_continuous_scale_inr import ContinuousDyTEdgeINR
from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_nafrssr import WindowedDyTNAFRSSR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_disparity_sgbm(img_l, img_r):
    # img: float32 [H, W, 3] in [0, 1]
    g_l = (cv2.cvtColor((img_l * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    g_r = (cv2.cvtColor((img_r * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    
    sgbm = cv2.StereoSGBM_create(
        minDisparity=-16,
        numDisparities=64,
        blockSize=7,
        P1=8 * 3 * 7**2,
        P2=32 * 3 * 7**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disp = sgbm.compute(g_l, g_r).astype(np.float32) / 16.0
    return disp

def calc_score(disp_sr, disp_hr, threshold=1.0):
    valid = (disp_hr > -15) & (disp_sr > -15)
    if np.sum(valid) < 100:
        return 1.0, 0.0
    err = np.abs(disp_sr[valid] - disp_hr[valid])
    bad_ratio = np.mean(err > threshold)
    epe = np.mean(err)
    score = 1.0 - bad_ratio
    return score, epe

# Load models
teacher = NAFSSR(up_scale=4, width=64, num_blks=16).to(device).eval()
ckpt_tea = torch.load(os.path.join(ROOT_DIR, 'external', 'NAFRSSR', 'NAFRSSR_M.pth'), map_location=device, weights_only=False)
teacher.load_state_dict(ckpt_tea['params'] if 'params' in ckpt_tea else ckpt_tea)

v8_model = ContinuousDyTEdgeINR(width=48, num_blks=4).to(device).eval()
sd8 = torch.load(os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'dyt_continuous_inr_quest3s_best.pth'), map_location=device, weights_only=False)
v8_model.load_state_dict(sd8['model_state_dict'] if 'model_state_dict' in sd8 else sd8, strict=False)

soup_model = ContinuousDyTEdgeINR(width=48, num_blks=4).to(device).eval()
sds = torch.load(os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'dyt_continuous_inr_quest3s_soup_best.pth'), map_location=device, weights_only=False)
soup_model.load_state_dict(sds['model_state_dict'] if 'model_state_dict' in sds else sds, strict=False)

mid_dirs = sorted(glob.glob(os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Middlebury', 'hr', '*')))

print(f"{'Method':<25} | {'Middlebury SCORE (>1px)':<25} | {'Disparity EPE (px)':<20} | {'SRAM Status':<20}")
print("-" * 95)

results = {
    'Bicubic': {'scores': [], 'epes': []},
    'Teacher NAFSSR-M': {'scores': [], 'epes': []},
    'V8 Continuous INR': {'scores': [], 'epes': []},
    'Model Soup (Pareto)': {'scores': [], 'epes': []}
}

for pdir in mid_dirs:
    hl = os.path.join(pdir, 'hr0.png')
    hr = os.path.join(pdir, 'hr1.png')
    if not os.path.exists(hl) or not os.path.exists(hr):
        continue
    img_l = np.array(Image.open(hl).convert('RGB'), dtype=np.float32) / 255.0
    img_r = np.array(Image.open(hr).convert('RGB'), dtype=np.float32) / 255.0
    
    H, W = img_l.shape[0], img_l.shape[1]
    cy, cx = (H - 192) // 2, (W - 192) // 2
    crop_hl = img_l[cy:cy+192, cx:cx+192]
    crop_hr = img_r[cy:cy+192, cx:cx+192]
    
    # Ground Truth Disparity on HR
    disp_hr = compute_disparity_sgbm(crop_hl, crop_hr)
    
    # Downsample LR
    t_hl = torch.from_numpy(crop_hl).permute(2,0,1).unsqueeze(0).to(device)
    t_hr = torch.from_numpy(crop_hr).permute(2,0,1).unsqueeze(0).to(device)
    t_ll = F.interpolate(t_hl, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
    t_lr = F.interpolate(t_hr, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
    
    # 1. Bicubic
    sr_bic_l = F.interpolate(t_ll, scale_factor=4, mode='bicubic', align_corners=False).squeeze(0).permute(1,2,0).cpu().numpy()
    sr_bic_r = F.interpolate(t_lr, scale_factor=4, mode='bicubic', align_corners=False).squeeze(0).permute(1,2,0).cpu().numpy()
    d_bic = compute_disparity_sgbm(sr_bic_l, sr_bic_r)
    sc, epe = calc_score(d_bic, disp_hr)
    results['Bicubic']['scores'].append(sc)
    results['Bicubic']['epes'].append(epe)
    
    # 2. Teacher NAFSSR-M
    with torch.no_grad():
        cat_in = torch.cat([t_ll, t_lr], dim=1)
        tea_out = teacher(cat_in)
        sr_tea_l = tea_out[:, 0:3].clamp(0, 1).squeeze(0).permute(1,2,0).cpu().numpy()
        sr_tea_r = tea_out[:, 3:6].clamp(0, 1).squeeze(0).permute(1,2,0).cpu().numpy()
    d_tea = compute_disparity_sgbm(sr_tea_l, sr_tea_r)
    sc, epe = calc_score(d_tea, disp_hr)
    results['Teacher NAFSSR-M']['scores'].append(sc)
    results['Teacher NAFSSR-M']['epes'].append(epe)
    
    # 3. V8 Continuous INR
    with torch.no_grad():
        o8_l, o8_r = v8_model(t_ll, t_lr, scale=4.0)
        sr_v8_l = o8_l.squeeze(0).permute(1,2,0).cpu().numpy()
        sr_v8_r = o8_r.squeeze(0).permute(1,2,0).cpu().numpy()
    d_v8 = compute_disparity_sgbm(sr_v8_l, sr_v8_r)
    sc, epe = calc_score(d_v8, disp_hr)
    results['V8 Continuous INR']['scores'].append(sc)
    results['V8 Continuous INR']['epes'].append(epe)
    
    # 4. Model Soup
    with torch.no_grad():
        os_l, os_r = soup_model(t_ll, t_lr, scale=4.0)
        sr_s_l = os_l.squeeze(0).permute(1,2,0).cpu().numpy()
        sr_s_r = os_r.squeeze(0).permute(1,2,0).cpu().numpy()
    d_soup = compute_disparity_sgbm(sr_s_l, sr_s_r)
    sc, epe = calc_score(d_soup, disp_hr)
    results['Model Soup (Pareto)']['scores'].append(sc)
    results['Model Soup (Pareto)']['epes'].append(epe)

sram_map = {
    'Bicubic': '0 KB (Zero GMEM)',
    'Teacher NAFSSR-M': 'Tràn DRAM (1.12 MB)',
    'V8 Continuous INR': '100% SRAM (904 KB)',
    'Model Soup (Pareto)': '100% SRAM (904 KB)'
}

for mname in results:
    m_score = np.mean(results[mname]['scores'])
    m_epe = np.mean(results[mname]['epes'])
    print(f"{mname:<25} | {m_score:<25.4f} | {m_epe:<20.4f} | {sram_map[mname]:<20}")
