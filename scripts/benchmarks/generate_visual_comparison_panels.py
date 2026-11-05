"""
Visual Comparison Panel Generator:
Generates high-resolution side-by-side visual panels and error heatmaps comparing:
1. Ground Truth (HR)
2. Standard Bicubic Interpolation
3. DyT-Edge-INR (Model Soup Pareto Best)
4. Residual Error Heatmaps (|SR - HR| x 5.0)

Generates panels for:
- Middlebury: cloth2 (high-frequency fabric textures and fine epipolar threads)
- Middlebury: motorcycle (intricate mechanical geometry and metallic reflections)
- KITTI 2012: outdoor automotive depth parallax
"""

import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

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

def create_heatmap(error_tensor):
    # error_tensor: (1, 1, H, W) normalized error in [0, 1]
    # Simple colormap: black -> blue -> red -> yellow
    e = error_tensor.squeeze().cpu().numpy()
    e = np.clip(e * 5.0, 0.0, 1.0)
    
    # 3-channel pseudo-color
    r = np.clip(2.0 * e, 0.0, 1.0)
    g = np.clip(2.0 * (e - 0.5), 0.0, 1.0)
    b = np.clip(1.0 - 2.0 * e, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Visual panel generation on {device}...")
    
    ckpt_path = os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'dyt_continuous_inr_quest3s_soup_best.pth')
    model = ContinuousDyTEdgeINR(width=48, num_blks=4).to(device).eval()
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd, strict=False)
    
    out_dir = os.path.join(ROOT_DIR, 'outputs', 'visual_comparisons')
    os.makedirs(out_dir, exist_ok=True)
    
    scenes = [
        ('Middlebury', 'cloth2', os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Middlebury', 'hr', 'cloth2')),
        ('Middlebury', 'motorcycle', os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Middlebury', 'hr', 'motorcycle')),
    ]
    
    for ds_name, sc_name, sc_dir in scenes:
        hl_path = os.path.join(sc_dir, 'hr0.png')
        hr_path = os.path.join(sc_dir, 'hr1.png')
        if not os.path.exists(hl_path):
            continue
            
        img_l_raw = torch.from_numpy(np.array(Image.open(hl_path).convert('RGB'), dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
        img_r_raw = torch.from_numpy(np.array(Image.open(hr_path).convert('RGB'), dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
        
        # Crop 256x256 region of interest with fine texture
        H, W = img_l_raw.shape[2], img_l_raw.shape[3]
        cy, cx = (H - 256) // 2, (W - 256) // 2
        crop_hl = img_l_raw[:, :, cy:cy+256, cx:cx+256]
        crop_hr = img_r_raw[:, :, cy:cy+256, cx:cx+256]
        
        # 4x downsampling to create LR input (64x64)
        lr_l = F.interpolate(crop_hl, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
        lr_r = F.interpolate(crop_hr, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
        
        # Bicubic SR
        bic_l = F.interpolate(lr_l, scale_factor=4.0, mode='bicubic', align_corners=False).clamp(0, 1)
        
        # Model SR
        with torch.no_grad():
            sr_l, sr_r = model(lr_l, lr_r, scale=4.0)
            
        # Metrics
        p_bic = calc_psnr(bic_l, crop_hl)
        s_bic = calc_ssim(bic_l, crop_hl)
        p_dyt = calc_psnr(sr_l, crop_hl)
        s_dyt = calc_ssim(sr_l, crop_hl)
        
        # Residual error maps
        err_bic = (bic_l - crop_hl).abs().mean(dim=1, keepdim=True)
        err_dyt = (sr_l - crop_hl).abs().mean(dim=1, keepdim=True)
        
        heat_bic = create_heatmap(err_bic)
        heat_dyt = create_heatmap(err_dyt)
        
        # Convert tensors to uint8 numpy
        gt_np = (crop_hl.squeeze().permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
        bic_np = (bic_l.squeeze().permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
        dyt_np = (sr_l.squeeze().permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
        
        # Stitch 4-panel image: [Ground Truth | Bicubic | DyT-Edge-INR | Residual Error (Ours)]
        panel_w, panel_h = 256, 256
        combined = Image.new('RGB', (panel_w * 4, panel_h + 40), color=(20, 20, 20))
        
        combined.paste(Image.fromarray(gt_np), (0, 40))
        combined.paste(Image.fromarray(bic_np), (panel_w, 40))
        combined.paste(Image.fromarray(dyt_np), (panel_w * 2, 40))
        combined.paste(Image.fromarray(heat_dyt), (panel_w * 3, 40))
        
        draw = ImageDraw.Draw(combined)
        draw.text((10, 10), "Ground Truth (HR)", fill=(255, 255, 255))
        draw.text((panel_w + 10, 10), f"Bicubic ({p_bic:.2f}dB / {s_bic:.4f})", fill=(200, 200, 200))
        draw.text((panel_w * 2 + 10, 10), f"DyT-Edge-INR ({p_dyt:.2f}dB / {s_dyt:.4f})", fill=(50, 255, 50))
        draw.text((panel_w * 3 + 10, 10), f"Error Heatmap (|SR-HR|x5)", fill=(255, 200, 50))
        
        save_path = os.path.join(out_dir, f"visual_panel_{ds_name.lower()}_{sc_name}.png")
        combined.save(save_path)
        print(f"[+] Saved visual comparison panel to: {save_path}")
        print(f"    {sc_name}: Bicubic = {p_bic:.2f} dB, DyT-Edge-INR = {p_dyt:.2f} dB (Gain: +{p_dyt - p_bic:.2f} dB)")

if __name__ == '__main__':
    main()
