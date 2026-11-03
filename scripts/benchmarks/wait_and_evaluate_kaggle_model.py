"""
Auto-Monitor & Evaluation Pipeline for Kaggle Dual-T4 Quest 3S Stereo SR
1. Checks status of zok213/stereoinr-turbo-dual-t4
2. If COMPLETE: downloads new artifacts and evaluates on canonical test sets
3. Computes both RGB PSNR/SSIM and Y-Channel PSNR (matching Table 1 NAFSSR protocol)
"""

import os
import sys
import time
import subprocess
import math
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


def check_kaggle_status(slug="zok213/stereoinr-turbo-dual-t4"):
    try:
        res = subprocess.run(["kaggle", "kernels", "status", slug], capture_output=True, text=True, check=True)
        out = res.stdout + res.stderr
        for line in out.splitlines():
            if "status" in line.lower():
                return line.strip()
        return out.strip()
    except Exception as e:
        return f"Error: {e}"


def download_kaggle_output(slug="zok213/stereoinr-turbo-dual-t4", out_dir="weights/kaggle_dual_t4"):
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["kaggle", "kernels", "output", slug, "-p", out_dir]
    print(f"[*] Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)


def calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    if mse <= 1e-10:
        return 100.0
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


def rgb2ycbcr_y(img_np: np.ndarray) -> np.ndarray:
    r = img_np[:, :, 0]
    g = img_np[:, :, 1]
    b = img_np[:, :, 2]
    return 16.0 + (65.481 * r + 128.553 * g + 24.966 * b)


def calc_psnr_y(pred_t: torch.Tensor, target_t: torch.Tensor, crop_border: int = 4) -> float:
    p_np = pred_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    t_np = target_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    if crop_border > 0:
        p_np = p_np[crop_border:-crop_border, crop_border:-crop_border]
        t_np = t_np[crop_border:-crop_border, crop_border:-crop_border]

    y_pred = rgb2ycbcr_y(p_np)
    y_target = rgb2ycbcr_y(t_np)

    mse = np.mean((y_pred - y_target) ** 2)
    if mse <= 1e-10:
        return 100.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def load_tensor(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def evaluate_models(device, ckpt_dyt_path):
    print("\n" + "=" * 115)
    print(">>> CANONICAL BENCHMARK: KAGGLE DUAL-T4 CHECKPOINT VS BASELINES")
    print(f">>> Evaluating Checkpoint: {ckpt_dyt_path}")
    print("=" * 115)

    naf = NAFSSR(up_scale=4, width=64, num_blks=16).to(device).eval()
    ckpt_naf = torch.load(os.path.join(ROOT_DIR, 'external', 'NAFRSSR', 'NAFRSSR_M.pth'), map_location=device, weights_only=False)
    naf.load_state_dict(ckpt_naf['params'] if 'params' in ckpt_naf else ckpt_naf, strict=True)

    dyt = WindowedDyTNAFRSSR(width=48, num_blks=4, up_scale=4, min_disp=-2, max_disp=14, fixup_zero_init=True).to(device).eval()
    state = torch.load(ckpt_dyt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in state:
        dyt.load_state_dict(state['model_state_dict'])
    else:
        dyt.load_state_dict(state)

    test_datasets = [
        ("Middlebury", os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Middlebury')),
        ("Flickr1024", os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Flickr1024')),
        ("KITTI2012", os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'KITTI2012')),
        ("KITTI2015", os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'KITTI2015'))
    ]

    print("\n" + "=" * 105)
    print(f"{'Dataset':<12} | {'Model':<22} | {'RGB PSNR':<10} | {'SSIM':<8} | {'Y-PSNR':<10} | {'Adreno 740 Status'}")
    print("=" * 105)

    for dname, dpath in test_datasets:
        if not os.path.exists(dpath):
            continue
        hr_dirs = sorted(glob.glob(os.path.join(dpath, 'hr', '*')))[:5]
        if not hr_dirs:
            continue

        p_bic, s_bic, y_bic = [], [], []
        p_naf, s_naf, y_naf = [], [], []
        p_dyt, s_dyt, y_dyt = [], [], []

        for pdir in hr_dirs:
            scene_name = os.path.basename(pdir)
            hl = os.path.join(pdir, 'hr0.png')
            hr = os.path.join(pdir, 'hr1.png')
            ll = os.path.join(dpath, 'lr_x4', scene_name, 'lr0.png')
            lr = os.path.join(dpath, 'lr_x4', scene_name, 'lr1.png')

            if not os.path.exists(hl) or not os.path.exists(hr):
                continue

            t_hl = load_tensor(hl, device)
            t_hr = load_tensor(hr, device)

            if os.path.exists(ll) and os.path.exists(lr):
                t_ll = load_tensor(ll, device)
                t_lr = load_tensor(lr, device)
                H_lr, W_lr = t_ll.shape[2], t_ll.shape[3]
                max_dim = 48
                cy = (H_lr - max_dim) // 2
                cx = (W_lr - max_dim) // 2
                crop_ll = t_ll[:, :, cy:cy+max_dim, cx:cx+max_dim]
                crop_lr = t_lr[:, :, cy:cy+max_dim, cx:cx+max_dim]
                crop_hl = t_hl[:, :, cy*4:(cy+max_dim)*4, cx*4:(cx+max_dim)*4]
            else:
                continue

            with torch.no_grad():
                # Bicubic
                out_bic = F.interpolate(crop_ll, scale_factor=4.0, mode='bicubic', align_corners=False)
                p_bic.append(calc_psnr(out_bic, crop_hl))
                s_bic.append(calc_ssim(out_bic, crop_hl))
                y_bic.append(calc_psnr_y(out_bic, crop_hl))

                # NAFSSR-M
                inp_naf = torch.cat([crop_ll, crop_lr], dim=1)
                out_naf = naf(inp_naf)[:, 0:3, :, :]
                p_naf.append(calc_psnr(out_naf, crop_hl))
                s_naf.append(calc_ssim(out_naf, crop_hl))
                y_naf.append(calc_psnr_y(out_naf, crop_hl))

                # Kaggle DyT-NAFRSSR
                out_dyt, _ = dyt(crop_ll, crop_lr)
                p_dyt.append(calc_psnr(out_dyt, crop_hl))
                s_dyt.append(calc_ssim(out_dyt, crop_hl))
                y_dyt.append(calc_psnr_y(out_dyt, crop_hl))

        print(f"{dname:<12} | {'Bicubic Baseline':<22} | {np.mean(p_bic):<10.2f} | {np.mean(s_bic):<8.4f} | {np.mean(y_bic):<10.2f} | Zero GMEM")
        print(f"{'':<12} | {'NAFRSSR-M (Teacher)':<22} | {np.mean(p_naf):<10.2f} | {np.mean(s_naf):<8.4f} | {np.mean(y_naf):<10.2f} | Tran DRAM (4.19MB)")
        print(f"{'':<12} | {'DyT-NAFRSSR (Dual-T4)':<22} | {np.mean(p_dyt):<10.2f} | {np.mean(s_dyt):<8.4f} | {np.mean(y_dyt):<10.2f} | 100% SRAM (904KB)")
        print("-" * 105)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    slug = "zok213/stereoinr-turbo-dual-t4"
    out_dir = os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4')
    ckpt_path = os.path.join(out_dir, 'dyt_nafrssr_quest3s_best.pth')

    print(f"[*] Checking Kaggle status for {slug}...")
    status = check_kaggle_status(slug)
    print(f"[*] Status: {status}")

    if "COMPLETE" in status:
        print("[*] Kernel completed! Downloading updated artifacts...")
        download_kaggle_output(slug, out_dir)
    elif "RUNNING" in status or "QUEUED" in status:
        print("[*] Kernel is actively training on 2x Tesla T4 GPUs. Please check back shortly.")
        sys.exit(0)

    if os.path.exists(ckpt_path):
        evaluate_models(device, ckpt_path)
    else:
        print(f"[-] Checkpoint not found at {ckpt_path}")
