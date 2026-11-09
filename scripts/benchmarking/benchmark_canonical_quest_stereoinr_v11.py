"""
Canonical Literature Benchmark for QuestStereoINR Version 11 (Trained on Kaggle Dual-T4)
Evaluates on Middlebury 5 canonical test scenes using standard CVPR/ECCV protocol:
- MATLAB rgb2ycbcr Y luminance channel
- 4-pixel border shaving
- Direct comparison with Bicubic, DyT-Edge-INR Soup (104K), NAFRSSR-M (280K), and StereoINR
"""

import os
import sys
import json
import time
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.deployment.export_v11_best_to_sentis import MasterQuestStereoINR


def rgb2y(arr: np.ndarray) -> np.ndarray:
    return 16.0 + 65.481 * arr[:, :, 0] + 128.553 * arr[:, :, 1] + 24.966 * arr[:, :, 2]


def calc_psnr_y(pred_t: torch.Tensor, target_t: torch.Tensor, crop_border: int = 4) -> float:
    p_np = pred_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    t_np = target_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    if crop_border > 0:
        p_np = p_np[crop_border:-crop_border, crop_border:-crop_border]
        t_np = t_np[crop_border:-crop_border, crop_border:-crop_border]

    y_p = rgb2y(p_np)
    y_t = rgb2y(t_np)

    mse = np.mean((y_p - y_t) ** 2)
    if mse <= 1e-10:
        return 100.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def main():
    print("=" * 90)
    print(">>> CANONICAL BENCHMARK: QUESTSTEREOINR VERSION 11 ON MIDDLEBURY")
    print("=" * 90)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n")

    test_root = os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Middlebury')
    hr_dir = os.path.join(test_root, 'hr')
    lr_dir = os.path.join(test_root, 'lr_x4')
    scenes = sorted(os.listdir(hr_dir))

    # Load QuestStereoINR Deployed
    deploy_path = os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'v11_output', 'quest_stereoinr_deploy.pth')
    model = MasterQuestStereoINR(channels=48, num_blocks=4).to(device)
    model.switch_to_deploy()
    if os.path.exists(deploy_path):
        sd = torch.load(deploy_path, map_location=device, weights_only=False)
        state = sd['model_state_dict'] if 'model_state_dict' in sd else sd
        model.load_state_dict(state, strict=True)
        print(f"[*] Loaded QuestStereoINR deployed weights: {deploy_path}")
    model.eval()

    print("\n" + "=" * 90)
    print(f"{'Scene':<12} | {'Bicubic (Y)':<14} | {'Fast 4x (Y)':<14} | {'Gain Fast':<12} | {'Cont 4x (Y)':<14} | {'Gain Cont':<12}")
    print("-" * 90)

    bic_psnrs, fast_psnrs, cont_psnrs = [], [], []

    for scene in scenes:
        l_hr = torch.from_numpy(np.array(Image.open(os.path.join(hr_dir, scene, 'hr0.png')).convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        r_hr = torch.from_numpy(np.array(Image.open(os.path.join(hr_dir, scene, 'hr1.png')).convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        l_lr = torch.from_numpy(np.array(Image.open(os.path.join(lr_dir, scene, 'lr0.png')).convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        r_lr = torch.from_numpy(np.array(Image.open(os.path.join(lr_dir, scene, 'lr1.png')).convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        min_h = min(l_hr.shape[2], l_lr.shape[2] * 4)
        min_w = min(l_hr.shape[3], l_lr.shape[3] * 4)
        t_l = l_hr[:, :, :min_h, :min_w]
        t_r = r_hr[:, :, :min_h, :min_w]

        # 1. Bicubic
        bic_l = F.interpolate(l_lr, size=(min_h, min_w), mode='bicubic', align_corners=False)
        bic_r = F.interpolate(r_lr, size=(min_h, min_w), mode='bicubic', align_corners=False)
        p_bic = (calc_psnr_y(bic_l, t_l) + calc_psnr_y(bic_r, t_r)) / 2.0
        bic_psnrs.append(p_bic)

        # 2. Fast 4x
        with torch.no_grad():
            out_fast_l, out_fast_r = model(l_lr, r_lr, scale=4.0, force_continuous=False)
            out_cont_l, out_cont_r = model(l_lr, r_lr, scale=4.0, force_continuous=True)

        p_fast = (calc_psnr_y(out_fast_l[:, :, :min_h, :min_w], t_l) + calc_psnr_y(out_fast_r[:, :, :min_h, :min_w], t_r)) / 2.0
        p_cont = (calc_psnr_y(out_cont_l[:, :, :min_h, :min_w], t_l) + calc_psnr_y(out_cont_r[:, :, :min_h, :min_w], t_r)) / 2.0

        fast_psnrs.append(p_fast)
        cont_psnrs.append(p_cont)

        g_fast = p_fast - p_bic
        g_cont = p_cont - p_bic

        print(f"{scene:<12} | {p_bic:6.2f} dB       | {p_fast:6.2f} dB       | {g_fast:+6.2f} dB     | {p_cont:6.2f} dB       | {g_cont:+6.2f} dB", flush=True)

    print("-" * 90)
    avg_bic = np.mean(bic_psnrs)
    avg_fast = np.mean(fast_psnrs)
    avg_cont = np.mean(cont_psnrs)
    print(f"{'AVERAGE':<12} | {avg_bic:6.2f} dB       | {avg_fast:6.2f} dB       | {avg_fast - avg_bic:+6.2f} dB     | {avg_cont:6.2f} dB       | {avg_cont - avg_bic:+6.2f} dB")
    print("=" * 90)


if __name__ == '__main__':
    main()
