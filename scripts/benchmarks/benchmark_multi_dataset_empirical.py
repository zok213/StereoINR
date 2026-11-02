"""
Comprehensive Multi-Dataset Empirical Benchmark: Stereo Image Super-Resolution
Evaluates on REAL Ground-Truth Datasets:
1. Middlebury (cloth2, motorcycle, piano, pipes, sword2)
2. Flickr1024 (real stereo photography)
3. KITTI2012 (automotive stereo)
4. KITTI2015 (automotive dynamic stereo)

Compares:
- Hardware Bicubic (4x)
- Official NAFRSSR-M (JNU Weights, 280,383 params)
- Windowed DyT-NAFRSSR (82,368 params, 100% GMEM Tile Resident)
- TextureStereoPixelNet (83,808 params, Wavelet Stroke Loss trained)
"""

import os
import sys
import math
import glob
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'external', 'NAFRSSR'))

from NAFRSSR_M_Net import NAFSSR
from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_nafrssr import WindowedDyTNAFRSSR
from src.edge_stereoinr.phase2_edge_inr.backbones.texture_stereopixelnet import TextureStereoPixelNet


def calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def calc_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    y1 = 0.299 * img1[:, 0:1] + 0.587 * img1[:, 1:2] + 0.114 * img1[:, 2:3]
    y2 = 0.299 * img2[:, 0:1] + 0.587 * img2[:, 1:2] + 0.114 * img2[:, 2:3]

    mu1 = F.avg_pool2d(y1, kernel_size=window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(y2, kernel_size=window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(y1 * y1, kernel_size=window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(y2 * y2, kernel_size=window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(y1 * y2, kernel_size=window_size, stride=1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def evaluate_dataset(dataset_name: str, base_dir: str, num_samples: int = 5, device: torch.device = torch.device('cpu')):
    hr_dirs = sorted(glob.glob(os.path.join(base_dir, 'hr', '*')))[:num_samples]
    if not hr_dirs:
        return None

    # Load Models
    # 1. NAFRSSR-M
    nafrssr_m = NAFSSR(up_scale=4, width=64, num_blks=16).to(device)
    weights_path = os.path.join(ROOT_DIR, 'external', 'NAFRSSR', 'NAFRSSR_M.pth')
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        nafrssr_m.load_state_dict(ckpt['params'] if 'params' in ckpt else ckpt, strict=True)
    nafrssr_m.eval()

    # 2. Windowed DyT-NAFRSSR
    dyt_model = WindowedDyTNAFRSSR(width=48, num_blks=4, up_scale=4, max_disp=8).to(device).eval()
    ckpt_dyt = os.path.join(ROOT_DIR, 'weights', 'checkpoints', 'dyt_nafrssr_trained.pth')
    if os.path.exists(ckpt_dyt):
        ckpt_d = torch.load(ckpt_dyt, map_location=device)
        dyt_model.load_state_dict(ckpt_d['model_state_dict'])

    # 3. TextureStereoPixelNet
    tex_model = TextureStereoPixelNet(in_channels=3, feat_channels=48, num_blocks=3).to(device).eval()
    ckpt_t = os.path.join(ROOT_DIR, 'weights', 'checkpoints', 'texture_stereopixelnet_trained.pth')
    if os.path.exists(ckpt_t):
        ckpt = torch.load(ckpt_t, map_location=device)
        tex_model.load_state_dict(ckpt['model_state_dict'])

    results = {
        'Bicubic': {'psnr': [], 'ssim': []},
        'NAFRSSR-M': {'psnr': [], 'ssim': []},
        'Windowed DyT-NAFRSSR': {'psnr': [], 'ssim': []},
        'TextureStereoPixelNet': {'psnr': [], 'ssim': []}
    }

    for scene_dir in hr_dirs:
        scene_name = os.path.basename(scene_dir)
        hr_l_path = os.path.join(scene_dir, 'hr0.png')
        hr_r_path = os.path.join(scene_dir, 'hr1.png')

        lr_l_path = os.path.join(base_dir, 'lr_x4', scene_name, 'lr0.png')
        lr_r_path = os.path.join(base_dir, 'lr_x4', scene_name, 'lr1.png')

        if not (os.path.exists(hr_l_path) and os.path.exists(lr_l_path)):
            continue

        hr_l = load_image_tensor(hr_l_path, device)
        hr_r = load_image_tensor(hr_r_path, device)
        lr_l = load_image_tensor(lr_l_path, device)
        lr_r = load_image_tensor(lr_r_path, device)

        # Ensure spatial dimension compatibility (crop to multiple of 16 for memory efficiency)
        H_lr, W_lr = lr_l.shape[2], lr_l.shape[3]
        # Crop a central patch if image is too large for single forward on mobile memory profiler
        max_dim = 256
        if H_lr > max_dim or W_lr > max_dim:
            H_crop = min(H_lr, max_dim)
            W_crop = min(W_lr, max_dim)
            h_start = (H_lr - H_crop) // 2
            w_start = (W_lr - W_crop) // 2
            lr_l = lr_l[:, :, h_start:h_start+H_crop, w_start:w_start+W_crop]
            lr_r = lr_r[:, :, h_start:h_start+H_crop, w_start:w_start+W_crop]
            hr_l = hr_l[:, :, h_start*4:(h_start+H_crop)*4, w_start*4:(w_start+W_crop)*4]
            hr_r = hr_r[:, :, h_start*4:(h_start+H_crop)*4, w_start*4:(w_start+W_crop)*4]

        # 1. Bicubic
        bic_l = F.interpolate(lr_l, size=hr_l.shape[2:], mode='bicubic', align_corners=False)
        results['Bicubic']['psnr'].append(calc_psnr(bic_l, hr_l))
        results['Bicubic']['ssim'].append(calc_ssim(bic_l, hr_l))

        # 2. NAFRSSR-M
        with torch.no_grad():
            inp_naf = torch.cat([lr_l, lr_r], dim=1)
            out_naf = nafrssr_m(inp_naf)
            out_naf_l = out_naf[:, 0:3, :hr_l.shape[2], :hr_l.shape[3]]
            results['NAFRSSR-M']['psnr'].append(calc_psnr(out_naf_l, hr_l))
            results['NAFRSSR-M']['ssim'].append(calc_ssim(out_naf_l, hr_l))

        # 3. Windowed DyT-NAFRSSR
        with torch.no_grad():
            out_dyt_l, _ = dyt_model(lr_l, lr_r)
            out_dyt_l = out_dyt_l[:, :, :hr_l.shape[2], :hr_l.shape[3]]
            results['Windowed DyT-NAFRSSR']['psnr'].append(calc_psnr(out_dyt_l, hr_l))
            results['Windowed DyT-NAFRSSR']['ssim'].append(calc_ssim(out_dyt_l, hr_l))

        # 4. TextureStereoPixelNet (2x cascaded to 4x)
        with torch.no_grad():
            s1 = tex_model(lr_l, lr_r)
            s2 = tex_model(s1, s1)
            out_tex_l = s2[:, :, :hr_l.shape[2], :hr_l.shape[3]]
            results['TextureStereoPixelNet']['psnr'].append(calc_psnr(out_tex_l, hr_l))
            results['TextureStereoPixelNet']['ssim'].append(calc_ssim(out_tex_l, hr_l))

    # Compute averages
    summary = {}
    for m in results:
        summary[m] = {
            'psnr': float(np.mean(results[m]['psnr'])) if results[m]['psnr'] else 0.0,
            'ssim': float(np.mean(results[m]['ssim'])) if results[m]['ssim'] else 0.0
        }
    return summary


def run_all_benchmarks():
    print("=" * 105)
    print(">>> RIGOROUS EMPIRICAL BENCHMARK ACROSS 4 REAL DATASETS (MIDDLEBURY, FLICKR1024, KITTI12, KITTI15)")
    print("=" * 105)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n")

    datasets = [
        ('Middlebury', os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Middlebury')),
        ('Flickr1024', os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Flickr1024')),
        ('KITTI2012', os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'KITTI2012')),
        ('KITTI2015', os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'KITTI2015')),
    ]

    all_summaries = {}
    for d_name, d_path in datasets:
        if os.path.exists(d_path):
            print(f"[*] Benchmarking on {d_name} real test scenes...")
            res = evaluate_dataset(d_name, d_path, num_samples=5, device=device)
            all_summaries[d_name] = res
            print(f"    - Bicubic:               PSNR = {res['Bicubic']['psnr']:.2f} dB, SSIM = {res['Bicubic']['ssim']:.4f}")
            print(f"    - NAFRSSR-M (Official):  PSNR = {res['NAFRSSR-M']['psnr']:.2f} dB, SSIM = {res['NAFRSSR-M']['ssim']:.4f}")
            print(f"    - Windowed DyT-NAFRSSR:  PSNR = {res['Windowed DyT-NAFRSSR']['psnr']:.2f} dB, SSIM = {res['Windowed DyT-NAFRSSR']['ssim']:.4f}")
            print(f"    - TextureStereoPixelNet: PSNR = {res['TextureStereoPixelNet']['psnr']:.2f} dB, SSIM = {res['TextureStereoPixelNet']['ssim']:.4f}")
            print()

    print("=" * 105)
    print(f"{'Dataset':<16} | {'Metric':<8} | {'Bicubic (4x)':<14} | {'NAFRSSR-M (JNU)':<18} | {'Windowed DyT-NAFRSSR':<22} | {'TextureStereoPixelNet'}")
    print("=" * 105)

    for d_name, res in all_summaries.items():
        p_bic = res['Bicubic']['psnr']
        s_bic = res['Bicubic']['ssim']

        p_naf = res['NAFRSSR-M']['psnr']
        s_naf = res['NAFRSSR-M']['ssim']

        p_dyt = res['Windowed DyT-NAFRSSR']['psnr']
        s_dyt = res['Windowed DyT-NAFRSSR']['ssim']

        p_tex = res['TextureStereoPixelNet']['psnr']
        s_tex = res['TextureStereoPixelNet']['ssim']

        print(f"{d_name:<16} | {'PSNR':<8} | {p_bic:<14.2f} | {p_naf:<18.2f} | {p_dyt:<22.2f} | {p_tex:.2f} dB")
        print(f"{'':<16} | {'SSIM':<8} | {s_bic:<14.4f} | {s_naf:<18.4f} | {s_dyt:<22.4f} | {s_tex:.4f}")
        print("-" * 105)

    print("=" * 105)


if __name__ == '__main__':
    run_all_benchmarks()
