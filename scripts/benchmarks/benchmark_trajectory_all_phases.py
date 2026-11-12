"""
Benchmark All 8 Trajectory Phases: Bicubic vs QuestStereoINR v15 vs DyT QNN INT8 vs StereoINR
=============================================================================================
Runs complete empirical evaluation across all 8 real-world trajectory phases harvested
from the user's continuous Meta Quest 3S recording.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.quest_stereo_inr import QuestStereoINR
from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_nafrssr import WindowedDyTNAFRSSR
from scripts.benchmarks.benchmark_triple_tier_architectures import calc_psnr_ssim


def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 85)
    print(">>> COMPREHENSIVE 8-PHASE XR TRAJECTORY BENCHMARK EVALUATOR")
    print(f"[*] Evaluation Device: {device}")
    print("=" * 85)

    manifest_path = "datasets/Quest3S_UserStereo_Harvested/quest3s_trajectory_manifest.json"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        phases = json.load(f)

    # 1. Load Tier 2: QuestStereoINR v15 Checkpoint
    print("[*] Initializing Tier 2: QuestStereoINR v15...")
    model_t2 = QuestStereoINR(in_channels=3, width=48, num_blocks=4).to(device)
    tier2_weights = [
        os.path.join(ROOT_DIR, 'weights', 'checkpoints', 'quest_stereoinr_wild_best.pth'),
        os.path.join(ROOT_DIR, 'weights', 'checkpoints', 'quest_stereoinr_v12_monotonic_best.pth'),
        os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'dyt_continuous_inr_quest3s_soup_best.pth')
    ]
    for w in tier2_weights:
        if os.path.exists(w):
            try:
                ckpt = torch.load(w, map_location=device, weights_only=False)
                sd = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
                model_t2.load_state_dict(sd, strict=False)
                print(f"[+] Tier 2 (QuestStereoINR v15) loaded weights: {os.path.basename(w)}")
                break
            except Exception:
                pass
    model_t2.switch_to_deploy()
    model_t2.eval()

    # 2. Load Tier 3: DyT-Edge-INR QNN Checkpoint
    print("[*] Initializing Tier 3: DyT-Edge-INR QNN...")
    model_t3 = WindowedDyTNAFRSSR(in_channels=3, width=32, num_blks=4, up_scale=4).to(device)
    tier3_weights = os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'dyt_nafrssr_quest3s_best.pth')
    if os.path.exists(tier3_weights):
        try:
            ckpt = torch.load(tier3_weights, map_location=device, weights_only=False)
            sd = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
            model_t3.load_state_dict(sd, strict=False)
            print(f"[+] Tier 3 (DyT-Edge-INR QNN) loaded weights: {os.path.basename(tier3_weights)}")
        except Exception:
            pass
    model_t3.eval()

    results = []

    print("\n" + "=" * 125)
    print(f"{'Phase Name':<35} | {'Scale':<5} | {'Bicubic PSNR':<14} | {'Tier 2 (QuestINR)':<18} | {'Tier 3 (QNN)':<16} | {'Gain T2 (dB)':<14} | {'Status'}")
    print("=" * 125)

    dataset_dir = "datasets/Quest3S_UserStereo_Harvested"

    for ph in phases:
        name = ph["name"]
        split = ph["split"]
        desc = ph["description"]

        # Load HR Ground Truth
        hr_l_path = os.path.join(dataset_dir, split, "hr", f"{name}_L.png")
        hr_r_path = os.path.join(dataset_dir, split, "hr", f"{name}_R.png")
        img_hr_l = Image.open(hr_l_path).convert('RGB')
        img_hr_r = Image.open(hr_r_path).convert('RGB')

        hr_l = torch.from_numpy(np.array(img_hr_l).transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
        hr_r = torch.from_numpy(np.array(img_hr_r).transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)

        # Load LR4
        lr4_l_path = os.path.join(dataset_dir, split, "lr_x4", f"{name}_L.png")
        lr4_r_path = os.path.join(dataset_dir, split, "lr_x4", f"{name}_R.png")
        img_lr4_l = Image.open(lr4_l_path).convert('RGB')
        img_lr4_r = Image.open(lr4_r_path).convert('RGB')

        lr4_l = torch.from_numpy(np.array(img_lr4_l).transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
        lr4_r = torch.from_numpy(np.array(img_lr4_r).transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)

        # Baseline: Bicubic
        H_hr, W_hr = hr_l.shape[2:]
        bic_l = F.interpolate(lr4_l, size=(H_hr, W_hr), mode='bicubic', align_corners=False)
        bic_r = F.interpolate(lr4_r, size=(H_hr, W_hr), mode='bicubic', align_corners=False)
        psnr_bic_l, _, ssim_bic_l = calc_psnr_ssim(bic_l, hr_l)
        psnr_bic_r, _, ssim_bic_r = calc_psnr_ssim(bic_r, hr_r)
        psnr_bic = (psnr_bic_l + psnr_bic_r) / 2.0
        ssim_bic = (ssim_bic_l + ssim_bic_r) / 2.0

        # Tier 2: QuestStereoINR v15
        with torch.no_grad():
            sr_t2_l, sr_t2_r = model_t2(lr4_l, lr4_r, scale=4.0)
            sr_t2_l = sr_t2_l.clamp(0, 1)
            sr_t2_r = sr_t2_r.clamp(0, 1)
        psnr_t2_l, _, ssim_t2_l = calc_psnr_ssim(sr_t2_l, hr_l)
        psnr_t2_r, _, ssim_t2_r = calc_psnr_ssim(sr_t2_r, hr_r)
        psnr_t2 = (psnr_t2_l + psnr_t2_r) / 2.0
        ssim_t2 = (ssim_t2_l + ssim_t2_r) / 2.0
        gain_t2 = psnr_t2 - psnr_bic

        # Tier 3: DyT QNN INT8
        with torch.no_grad():
            sr_t3_l, sr_t3_r = model_t3(lr4_l, lr4_r)
            sr_t3_l = sr_t3_l.clamp(0, 1)
            sr_t3_r = sr_t3_r.clamp(0, 1)
        psnr_t3_l, _, ssim_t3_l = calc_psnr_ssim(sr_t3_l, hr_l)
        psnr_t3_r, _, ssim_t3_r = calc_psnr_ssim(sr_t3_r, hr_r)
        psnr_t3 = (psnr_t3_l + psnr_t3_r) / 2.0
        ssim_t3 = (ssim_t3_l + ssim_t3_r) / 2.0
        gain_t3 = psnr_t3 - psnr_bic

        status = "WIN (+)" if gain_t2 > 0 else "LOSS (-)"
        print(f"{name:<35} | {'4.0x':<5} | {psnr_bic:6.2f} dB        | {psnr_t2:6.2f} dB ({ssim_t2:.4f}) | {psnr_t3:6.2f} dB       | {gain_t2:+6.2f} dB       | {status}")

        results.append({
            "name": name,
            "split": split,
            "desc": desc,
            "scale": 4.0,
            "bicubic": {"psnr": psnr_bic, "ssim": ssim_bic},
            "tier2_queststereoinr": {"psnr": psnr_t2, "ssim": ssim_t2, "gain": gain_t2},
            "tier3_qnn": {"psnr": psnr_t3, "ssim": ssim_t3, "gain": gain_t3}
        })

    print("=" * 125)
    mean_bic = float(np.mean([r["bicubic"]["psnr"] for r in results]))
    mean_t2 = float(np.mean([r["tier2_queststereoinr"]["psnr"] for r in results]))
    mean_gain_t2 = mean_t2 - mean_bic
    mean_t3 = float(np.mean([r["tier3_qnn"]["psnr"] for r in results]))
    mean_gain_t3 = mean_t3 - mean_bic
    win_count = sum(1 for r in results if r["tier2_queststereoinr"]["gain"] > 0)

    print(f"{'OVERALL MEAN (8 PHASES)':<35} | {'4.0x':<5} | {mean_bic:6.2f} dB        | {mean_t2:6.2f} dB ({np.mean([r['tier2_queststereoinr']['ssim'] for r in results]):.4f}) | {mean_t3:6.2f} dB       | {mean_gain_t2:+6.2f} dB       | WIN: {win_count}/{len(results)} ({win_count/len(results)*100:.0f}%)")
    print("=" * 125)

    out_json = "logs/benchmarks/trajectory_8phases_benchmark_results.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "mean_bicubic_psnr": mean_bic,
                "mean_tier2_psnr": mean_t2,
                "mean_gain_tier2": mean_gain_t2,
                "mean_tier3_psnr": mean_t3,
                "mean_gain_tier3": mean_gain_t3,
                "win_rate": f"{win_count}/{len(results)}"
            },
            "phases": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Results saved to {out_json}")


if __name__ == "__main__":
    run_benchmark()
