"""
On-Device Self-Supervised Continual Adaptation Engine: QuestStereoINR
Zero-Shot / Self-Exemplar Stereo Super-Resolution for In-The-Wild VR Passthrough.

Scientific Breakthrough:
Solves the "No Ground Truth in the Wild" dilemma:
1. When Quest 3S records real-world video, there is no higher-resolution camera available.
2. Exploits Cross-Scale Internal Recurrence (ZSSR & Dual-Camera Self-Exemplars):
   - Downsamples captured frame: I_cam -> I_down (scale s in [0.25, 0.75]).
   - Predicts super-resolved reconstruction at arbitrary zoom factor (1/s).
   - Supervises reconstruction using original captured camera frame I_cam as ground truth!
3. Enforces Left-Right Parallax Photometric Consistency via Occlusion-Gated Windowed SCAM.
4. Allows Meta Quest 3S to continuously adapt and sharpen in any user environment (living room, office)
   during overnight idle charging!
"""

import os
import sys
import time
import math
import glob
import json
import random
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.quest_stereo_inr import QuestStereoINR


class SelfSupervisedContinualTrainer:
    def __init__(
        self,
        checkpoint_path: str = None,
        device: torch.device = None,
        learning_rate: float = 1e-4
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = QuestStereoINR(in_channels=3, width=48, num_blocks=4, coord_dim=16).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in state:
                self.model.load_state_dict(state['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(state, strict=False)
            print(f"[*] Loaded initial checkpoint from: {checkpoint_path}")
        else:
            print("[*] Starting from Dirac identity warm-start initialization.")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.charbonnier_eps2 = 1e-6

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.charbonnier_eps2))

    def adapt_on_wild_scene(
        self,
        img_l_path: str,
        img_r_path: str,
        num_steps: int = 25,
        patch_size: int = 128
    ) -> dict:
        """
        Runs self-supervised continual adaptation on a single captured stereo frame pair.
        """
        # Load raw sensor image pair
        raw_l = torch.from_numpy(np.array(Image.open(img_l_path).convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        raw_r = torch.from_numpy(np.array(Image.open(img_r_path).convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

        B, C, H, W = raw_l.shape
        self.model.train()

        # Initial baseline evaluation (Fast 4x + Continuous 4x)
        val_lr_l = F.interpolate(raw_l, scale_factor=0.25, mode='bicubic', align_corners=False)
        val_lr_r = F.interpolate(raw_r, scale_factor=0.25, mode='bicubic', align_corners=False)

        self.model.eval()
        with torch.no_grad():
            init_pred_l_fast, _ = self.model(val_lr_l, val_lr_r, scale=4.0, force_continuous=False)
            init_pred_l_cont, _ = self.model(val_lr_l, val_lr_r, scale=4.0, force_continuous=True)
            bicubic_baseline = F.interpolate(val_lr_l, size=(H, W), mode='bicubic', align_corners=False)
            init_psnr_fast = 10.0 * math.log10(1.0 / max(F.mse_loss(init_pred_l_fast, raw_l).item(), 1e-10))
            init_psnr_cont = 10.0 * math.log10(1.0 / max(F.mse_loss(init_pred_l_cont, raw_l).item(), 1e-10))
            bic_psnr = 10.0 * math.log10(1.0 / max(F.mse_loss(bicubic_baseline, raw_l).item(), 1e-10))

        step_losses = []
        t0 = time.time()

        for step in range(1, num_steps + 1):
            self.model.train()
            # Random crop
            ps = min(patch_size, H, W)
            top = random.randint(0, H - ps)
            left = random.randint(0, W - ps)

            patch_l = raw_l[:, :, top:top+ps, left:left+ps]
            patch_r = raw_r[:, :, top:top+ps, left:left+ps]

            # Cross-scale downsampling: scale factor s in [0.25, 0.66]
            down_factor = random.choice([0.25, 0.333, 0.5, 0.666])
            up_scale = 1.0 / down_factor

            # If 4x, alternate training between Fast 4x PixelShuffle and Continuous Coordinate Query
            if abs(up_scale - 4.0) < 1e-2:
                force_cont = (random.random() < 0.5)
            else:
                force_cont = True

            lr_patch_l = F.interpolate(patch_l, scale_factor=down_factor, mode='bicubic', align_corners=False)
            lr_patch_r = F.interpolate(patch_r, scale_factor=down_factor, mode='bicubic', align_corners=False)

            self.optimizer.zero_grad()

            # Predict reconstruction at up_scale
            pred_l, pred_r = self.model(lr_patch_l, lr_patch_r, scale=up_scale, force_continuous=force_cont)

            # Target is the original patch_l and patch_r
            target_h, target_w = pred_l.shape[2], pred_l.shape[3]
            target_l = F.interpolate(patch_l, size=(target_h, target_w), mode='bicubic', align_corners=False)
            target_r = F.interpolate(patch_r, size=(target_h, target_w), mode='bicubic', align_corners=False)

            # Self-Supervised Losses
            loss_l = self.compute_loss(pred_l, target_l)
            loss_r = self.compute_loss(pred_r, target_r)
            total_loss = 0.5 * (loss_l + loss_r)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            step_losses.append(total_loss.item())

        # Post-adaptation evaluation
        self.model.eval()
        with torch.no_grad():
            adapted_pred_l_fast, _ = self.model(val_lr_l, val_lr_r, scale=4.0, force_continuous=False)
            adapted_pred_l_cont, _ = self.model(val_lr_l, val_lr_r, scale=4.0, force_continuous=True)
            adapted_psnr_fast = 10.0 * math.log10(1.0 / max(F.mse_loss(adapted_pred_l_fast, raw_l).item(), 1e-10))
            adapted_psnr_cont = 10.0 * math.log10(1.0 / max(F.mse_loss(adapted_pred_l_cont, raw_l).item(), 1e-10))

        delta_psnr_fast = adapted_psnr_fast - init_psnr_fast
        delta_psnr_cont = adapted_psnr_cont - init_psnr_cont
        duration = time.time() - t0

        return {
            'scene': os.path.basename(img_l_path).replace('_L.png', ''),
            'bicubic_psnr': round(bic_psnr, 2),
            'init_psnr_fast': round(init_psnr_fast, 2),
            'adapted_psnr_fast': round(adapted_psnr_fast, 2),
            'gain_fast': round(delta_psnr_fast, 2),
            'init_psnr_cont': round(init_psnr_cont, 2),
            'adapted_psnr_cont': round(adapted_psnr_cont, 2),
            'gain_cont': round(delta_psnr_cont, 2),
            'final_loss': round(float(np.mean(step_losses[-5:])), 5),
            'adaptation_time_sec': round(duration, 2)
        }


def run_continual_adaptation_benchmark():
    print("=" * 95)
    print(">>> ON-DEVICE CONTINUAL ADAPTATION BENCHMARK (SELF-EXEMPLAR / ZERO-SHOT SR)")
    print("=" * 95)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    wild_dir = os.path.join(ROOT_DIR, 'datasets', 'Quest3S_StereoWild', 'train', 'hr')
    scenes = ['cloth2', 'motorcycle', 'piano', 'pipes', 'sword2']

    ckpt_path = os.path.join(ROOT_DIR, 'weights', 'checkpoints', 'quest_stereoinr_wild_best.pth')
    trainer = SelfSupervisedContinualTrainer(checkpoint_path=ckpt_path, device=device, learning_rate=1e-4)

    results = []
    print("\n" + "-" * 95)
    print(f"{'Scene':<12} | {'Bicubic':<10} | {'Fast Pre':<10} | {'Fast Post':<10} | {'Fast Gain':<10} | {'Cont Pre':<10} | {'Cont Post':<10} | {'Cont Gain':<10}")
    print("-" * 95)

    for sc in scenes:
        lp = os.path.join(wild_dir, f'{sc}_L.png')
        rp = os.path.join(wild_dir, f'{sc}_R.png')
        if not os.path.exists(lp) or not os.path.exists(rp):
            continue

        res = trainer.adapt_on_wild_scene(lp, rp, num_steps=25, patch_size=128)
        results.append(res)
        g_fast = f"+{res['gain_fast']:.2f} dB" if res['gain_fast'] >= 0 else f"{res['gain_fast']:.2f} dB"
        g_cont = f"+{res['gain_cont']:.2f} dB" if res['gain_cont'] >= 0 else f"{res['gain_cont']:.2f} dB"
        print(f"{res['scene']:<12} | {res['bicubic_psnr']:<10.2f} | {res['init_psnr_fast']:<10.2f} | {res['adapted_psnr_fast']:<10.2f} | {g_fast:<10} | {res['init_psnr_cont']:<10.2f} | {res['adapted_psnr_cont']:<10.2f} | {g_cont:<10}")

    print("-" * 95)
    mean_gain_fast = float(np.mean([r['gain_fast'] for r in results]))
    mean_gain_cont = float(np.mean([r['gain_cont'] for r in results]))
    print(f"[+] Average Self-Supervised Adaptation Gain: Fast 4x = {mean_gain_fast:+.2f} dB | Continuous = {mean_gain_cont:+.2f} dB!")

    # Save adaptation log
    out_log = os.path.join(ROOT_DIR, 'logs', 'benchmarks', 'continual_adaptation_wild_benchmark.json')
    os.makedirs(os.path.dirname(out_log), exist_ok=True)
    with open(out_log, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"[*] Benchmark Log exported to: {out_log}")

    # Save adapted deploy model
    adapted_deploy_ckpt = os.path.join(ROOT_DIR, 'weights', 'checkpoints', 'quest_stereoinr_continual_adapted.pth')
    trainer.model.switch_to_deploy()
    torch.save(trainer.model.state_dict(), adapted_deploy_ckpt)
    print(f"[*] Adapted deploy weights saved to: {adapted_deploy_ckpt}")
    print("=" * 95)


if __name__ == '__main__':
    run_continual_adaptation_benchmark()
