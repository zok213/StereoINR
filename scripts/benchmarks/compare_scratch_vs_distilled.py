"""
Head-to-head empirical comparison:
1. Bicubic Baseline
2. Kaggle Dual-T4 Scratch Model (20 epochs, Flickr1024)
3. Distilled Student Model (DyT-NAFRSSR, cached teacher guidance)
4. NAFSSR-M (Canonical Teacher, 280,383 params)
"""

import os
import sys
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'external', 'NAFRSSR'))

from NAFRSSR_M_Net import NAFSSR
from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_nafrssr import WindowedDyTNAFRSSR


def calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    if mse <= 1e-10:
        return 100.0
    import math
    return 10.0 * math.log10(1.0 / mse)


def calc_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
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


def load_tensor(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 110)
    print(">>> CRITICAL EMPIRICAL COMPARISON: SCRATCH TRAINING VS TEACHER DISTILLATION")
    print(f">>> Device: {device}")
    print("=" * 110)

    # 1. NAFSSR-M
    naf = NAFSSR(up_scale=4, width=64, num_blks=16).to(device).eval()
    ckpt_naf = torch.load('external/NAFRSSR/NAFRSSR_M.pth', map_location=device, weights_only=False)
    naf.load_state_dict(ckpt_naf['params'] if 'params' in ckpt_naf else ckpt_naf)

    # 2. Kaggle Scratch Model
    model_scratch = WindowedDyTNAFRSSR(width=48, num_blks=4, up_scale=4, min_disp=-2, max_disp=14).to(device).eval()
    ckpt_scratch = 'weights/kaggle_dual_t4/dyt_nafrssr_quest3s_best.pth'
    if os.path.exists(ckpt_scratch):
        s = torch.load(ckpt_scratch, map_location=device, weights_only=False)
        model_scratch.load_state_dict(s['model_state_dict'] if 'model_state_dict' in s else s)

    # 3. Local Distilled Model
    model_dist = WindowedDyTNAFRSSR(width=48, num_blks=4, up_scale=4, min_disp=-2, max_disp=14).to(device).eval()
    ckpt_dist = 'weights/checkpoints/dyt_nafrssr_distilled.pth'
    if os.path.exists(ckpt_dist):
        d = torch.load(ckpt_dist, map_location=device, weights_only=False)
        model_dist.load_state_dict(d['model_state_dict'] if 'model_state_dict' in d else d)

    datasets = [
        ("Middlebury", "datasets/StereoSR/test/Middlebury"),
        ("Flickr1024", "datasets/StereoSR/test/Flickr1024"),
        ("KITTI2012", "datasets/StereoSR/test/KITTI2012"),
        ("KITTI2015", "datasets/StereoSR/test/KITTI2015"),
    ]

    print(f"\n{'Dataset':<12} | {'Metric':<6} | {'Bicubic':<10} | {'Scratch (Kaggle)':<16} | {'Distilled (Ours)':<16} | {'NAFSSR-M (Teacher)'}")
    print("-" * 110)

    for dname, dpath in datasets:
        if not os.path.exists(dpath):
            continue
        hr_dirs = sorted(glob.glob(os.path.join(dpath, 'hr', '*')))[:5]
        if not hr_dirs:
            continue

        p_bic, s_bic = [], []
        p_scr, s_scr = [], []
        p_dst, s_dst = [], []
        p_naf, s_naf = [], []

        for pdir in hr_dirs:
            sname = os.path.basename(pdir)
            hl = os.path.join(pdir, 'hr0.png')
            hr = os.path.join(pdir, 'hr1.png')
            ll = os.path.join(dpath, 'lr_x4', sname, 'lr0.png')
            lr = os.path.join(dpath, 'lr_x4', sname, 'lr1.png')

            if not os.path.exists(hl) or not os.path.exists(hr) or not os.path.exists(ll) or not os.path.exists(lr):
                continue

            t_hl = load_tensor(hl, device)
            t_hr = load_tensor(hr, device)
            t_ll = load_tensor(ll, device)
            t_lr = load_tensor(lr, device)

            H_lr, W_lr = t_ll.shape[2], t_ll.shape[3]
            max_dim = 48
            cy = (H_lr - max_dim) // 2
            cx = (W_lr - max_dim) // 2
            c_ll = t_ll[:, :, cy:cy+max_dim, cx:cx+max_dim]
            c_lr = t_lr[:, :, cy:cy+max_dim, cx:cx+max_dim]
            c_hl = t_hl[:, :, cy*4:(cy+max_dim)*4, cx*4:(cx+max_dim)*4]

            with torch.no_grad():
                out_bic = F.interpolate(c_ll, scale_factor=4.0, mode='bicubic', align_corners=False)
                out_scr, _ = model_scratch(c_ll, c_lr)
                out_dst, _ = model_dist(c_ll, c_lr)
                out_naf = naf(torch.cat([c_ll, c_lr], dim=1))[:, 0:3, :, :]

                p_bic.append(calc_psnr(out_bic, c_hl))
                s_bic.append(calc_ssim(out_bic, c_hl))

                p_scr.append(calc_psnr(out_scr, c_hl))
                s_scr.append(calc_ssim(out_scr, c_hl))

                p_dst.append(calc_psnr(out_dst, c_hl))
                s_dst.append(calc_ssim(out_dst, c_hl))

                p_naf.append(calc_psnr(out_naf, c_hl))
                s_naf.append(calc_ssim(out_naf, c_hl))

        print(f"{dname:<12} | PSNR   | {np.mean(p_bic):<10.2f} | {np.mean(p_scr):<16.2f} | {np.mean(p_dst):<16.2f} | {np.mean(p_naf):<10.2f}")
        print(f"{'':<12} | SSIM   | {np.mean(s_bic):<10.4f} | {np.mean(s_scr):<16.4f} | {np.mean(s_dst):<16.4f} | {np.mean(s_naf):<10.4f}")
        print("-" * 110)


if __name__ == '__main__':
    main()
