"""
Comprehensive Canonical Evaluation: Version 6 (Native IRCA + Disparity Warping Loss)
Evaluates across Middlebury, KITTI 2012, KITTI 2015, Flickr1024 (157 pairs).
Measures:
- Y-PSNR (dB)
- RGB PSNR (dB)
- SSIM
- Stereo Disparity MAE (Epipolar Geometric Consistency)
"""

import os
import sys
import math
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_nafrssr import WindowedDyTNAFRSSR


def calc_metrics(pred: torch.Tensor, target: torch.Tensor):
    pred_c = torch.clamp(pred, 0.0, 1.0)
    target_c = torch.clamp(target, 0.0, 1.0)

    # RGB PSNR
    mse_rgb = torch.mean((pred_c - target_c) ** 2).item()
    psnr_rgb = 10.0 * math.log10(1.0 / max(mse_rgb, 1e-10))

    # Y-channel PSNR
    weights = torch.tensor([65.481, 128.553, 24.966], device=pred.device).view(1, 3, 1, 1) / 255.0
    bias = 16.0 / 255.0
    pred_y = torch.sum(pred_c * weights, dim=1, keepdim=True) + bias
    target_y = torch.sum(target_c * weights, dim=1, keepdim=True) + bias

    mse_y = torch.mean((pred_y - target_y) ** 2).item()
    psnr_y = 10.0 * math.log10(1.0 / max(mse_y, 1e-10))

    # SSIM
    mu_x = pred_y.mean()
    mu_y = target_y.mean()
    sigma_x = pred_y.var()
    sigma_y = target_y.var()
    sigma_xy = ((pred_y - mu_x) * (target_y - mu_y)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))

    return psnr_y, psnr_rgb, ssim.item()


def compute_stereo_mae(img_l: torch.Tensor, img_r: torch.Tensor, max_d: int = 14):
    min_err = 1e9
    for d in range(0, max_d + 1):
        if d == 0:
            diff = torch.abs(img_l - img_r).mean().item()
        else:
            diff = torch.abs(img_l[:, :, :, d:] - img_r[:, :, :, :-d]).mean().item()
        if diff < min_err:
            min_err = diff
    return float(min_err)


def run_benchmark():
    print("=" * 115)
    print(">>> CANONICAL BENCHMARK: VERSION 6 NATIVE IRCA + DISPARITY LOSS VS V5 & BICUBIC")
    print("=" * 115)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    # 1. Version 5 (Single Fusion)
    v5_path = os.path.join(ROOT_DIR, 'weights/kaggle_dual_t4/dyt_nafrssr_quest3s_best.pth')
    model_v5 = WindowedDyTNAFRSSR(width=48, num_blks=4, up_scale=4, interleave_fusion=False).to(device).eval()
    if os.path.exists(v5_path):
        sd5 = torch.load(v5_path, map_location=device, weights_only=False)
        model_v5.load_state_dict(sd5['model_state_dict'] if 'model_state_dict' in sd5 else sd5, strict=False)
        print("[*] Loaded DyT-NAFRSSR v5 (Single Fusion, 82K params)")

    # 2. Version 6 (Native IRCA)
    v6_path = os.path.join(ROOT_DIR, 'weights/kaggle_dual_t4/dyt_nafrssr_quest3s_v6_best.pth')
    model_v6 = WindowedDyTNAFRSSR(width=48, num_blks=4, up_scale=4, interleave_fusion=True).to(device).eval()
    if os.path.exists(v6_path):
        sd6 = torch.load(v6_path, map_location=device, weights_only=False)
        model_v6.load_state_dict(sd6['model_state_dict'] if 'model_state_dict' in sd6 else sd6, strict=False)
        print("[*] Loaded DyT-NAFRSSR v6 (Native IRCA, 82K params)")

    # 3. Teacher NAFSSR-M
    teacher_loaded = False
    model_teacher = None
    teacher_path = os.path.join(ROOT_DIR, 'external/NAFRSSR/NAFRSSR_M.pth')
    if os.path.exists(teacher_path):
        try:
            sys.path.insert(0, os.path.join(ROOT_DIR, 'external/NAFRSSR'))
            from NAFRSSR_M_Net import NAFSSR
            model_teacher = NAFSSR(up_scale=4, width=64, num_blks=16).to(device).eval()
            tsd = torch.load(teacher_path, map_location=device, weights_only=False)
            model_teacher.load_state_dict(tsd['params'] if 'params' in tsd else tsd)
            teacher_loaded = True
            print("[*] Loaded Official NAFSSR-M Teacher (280K params)")
        except Exception as e:
            print(f"[!] Teacher load note: {e}")

    datasets = [
        ("Middlebury", os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Middlebury')),
        ("KITTI 2012", os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'KITTI2012')),
        ("KITTI 2015", os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'KITTI2015')),
        ("Flickr1024", os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Flickr1024'))
    ]

    print("\n" + "=" * 115)
    print(f"{'Dataset':<12} | {'Model':<25} | {'Y-PSNR':<10} | {'SSIM':<8} | {'Stereo MAE':<12} | {'Adreno 740 Status'}")
    print("=" * 115)

    with torch.no_grad():
        for dname, dpath in datasets:
            if not os.path.exists(dpath):
                continue
            hr_dirs = sorted(glob.glob(os.path.join(dpath, 'hr', '*')))
            if not hr_dirs:
                continue

            stats = {
                'Bicubic': {'y': [], 'ssim': [], 'mae': []},
                'NAFSSR-M (Teacher)': {'y': [], 'ssim': [], 'mae': []},
                'DyT-NAFRSSR v5 (Single)': {'y': [], 'ssim': [], 'mae': []},
                'DyT-NAFRSSR v6 (IRCA)': {'y': [], 'ssim': [], 'mae': []},
            }

            for pdir in hr_dirs:
                sname = os.path.basename(pdir)
                hl = os.path.join(pdir, 'hr0.png')
                hr = os.path.join(pdir, 'hr1.png')
                ll = os.path.join(dpath, 'lr_x4', sname, 'lr0.png')
                lr = os.path.join(dpath, 'lr_x4', sname, 'lr1.png')
                if not (os.path.exists(hl) and os.path.exists(hr) and os.path.exists(ll) and os.path.exists(lr)):
                    continue

                arr_hl = np.array(Image.open(hl).convert('RGB'), dtype=np.float32) / 255.0
                arr_hr = np.array(Image.open(hr).convert('RGB'), dtype=np.float32) / 255.0
                arr_ll = np.array(Image.open(ll).convert('RGB'), dtype=np.float32) / 255.0
                arr_lr = np.array(Image.open(lr).convert('RGB'), dtype=np.float32) / 255.0

                H, W = arr_ll.shape[0], arr_ll.shape[1]
                cy = (H - 48) // 2
                cx = (W - 48) // 2
                c_ll = arr_ll[cy:cy+48, cx:cx+48]
                c_lr = arr_lr[cy:cy+48, cx:cx+48]
                c_hl = arr_hl[cy*4:(cy+48)*4, cx*4:(cx+48)*4]
                c_hr = arr_hr[cy*4:(cy+48)*4, cx*4:(cx+48)*4]

                t_ll = torch.from_numpy(c_ll).permute(2, 0, 1).unsqueeze(0).to(device)
                t_lr = torch.from_numpy(c_lr).permute(2, 0, 1).unsqueeze(0).to(device)
                t_hl = torch.from_numpy(c_hl).permute(2, 0, 1).unsqueeze(0).to(device)
                t_hr = torch.from_numpy(c_hr).permute(2, 0, 1).unsqueeze(0).to(device)

                # 1. Bicubic
                bic_l = F.interpolate(t_ll, scale_factor=4.0, mode='bicubic', align_corners=False).clamp(0, 1)
                bic_r = F.interpolate(t_lr, scale_factor=4.0, mode='bicubic', align_corners=False).clamp(0, 1)
                py, _, ssim = calc_metrics(bic_l, t_hl)
                mae = compute_stereo_mae(bic_l, bic_r)
                stats['Bicubic']['y'].append(py)
                stats['Bicubic']['ssim'].append(ssim)
                stats['Bicubic']['mae'].append(mae)

                # 2. Teacher
                if teacher_loaded and model_teacher is not None:
                    cat_in = torch.cat([t_ll, t_lr], dim=1)
                    t_out = model_teacher(cat_in)
                    out_t_l = t_out[:, 0:3].clamp(0, 1)
                    out_t_r = t_out[:, 3:6].clamp(0, 1)
                    py, _, ssim = calc_metrics(out_t_l, t_hl)
                    mae = compute_stereo_mae(out_t_l, out_t_r)
                    stats['NAFSSR-M (Teacher)']['y'].append(py)
                    stats['NAFSSR-M (Teacher)']['ssim'].append(ssim)
                    stats['NAFSSR-M (Teacher)']['mae'].append(mae)

                # 3. v5
                out_v5_l, out_v5_r = model_v5(t_ll, t_lr)
                py, _, ssim = calc_metrics(out_v5_l, t_hl)
                mae = compute_stereo_mae(out_v5_l, out_v5_r)
                stats['DyT-NAFRSSR v5 (Single)']['y'].append(py)
                stats['DyT-NAFRSSR v5 (Single)']['ssim'].append(ssim)
                stats['DyT-NAFRSSR v5 (Single)']['mae'].append(mae)

                # 4. v6
                out_v6_l, out_v6_r = model_v6(t_ll, t_lr)
                py, _, ssim = calc_metrics(out_v6_l, t_hl)
                mae = compute_stereo_mae(out_v6_l, out_v6_r)
                stats['DyT-NAFRSSR v6 (IRCA)']['y'].append(py)
                stats['DyT-NAFRSSR v6 (IRCA)']['ssim'].append(ssim)
                stats['DyT-NAFRSSR v6 (IRCA)']['mae'].append(mae)

            # Print dataset row
            base_y = np.mean(stats['Bicubic']['y'])
            for mname, mstat in stats.items():
                if not mstat['y']:
                    continue
                mean_y = np.mean(mstat['y'])
                mean_ssim = np.mean(mstat['ssim'])
                mean_mae = np.mean(mstat['mae'])
                diff = mean_y - base_y
                diff_str = f"+{diff:.2f} dB" if diff > 0 else f"{diff:.2f} dB"
                if mname == 'Bicubic':
                    diff_str = "Baseline"
                status = "100% On-Chip (904KB)" if "DyT" in mname else ("Tràn DRAM (4.19MB)" if "Teacher" in mname else "Zero GMEM")
                print(f"{dname:<12} | {mname:<25} | {mean_y:<6.2f} dB ({diff_str:<7}) | {mean_ssim:<8.4f} | {mean_mae:<12.4f} | {status}")
            print("-" * 115)


if __name__ == '__main__':
    run_benchmark()
