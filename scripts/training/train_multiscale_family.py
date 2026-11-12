"""
Training Script: Multi-Scale Architectural Family (Medium & Large Tiers)
========================================================================
Unleashes model capacity beyond the ultra-compact 125K baseline:
- Medium Tier: Width 80, Num Blocks 8  (~448K params, 3.6x capacity)
- Large Tier:  Width 128, Num Blocks 16 (~1.78M params, 14.2x capacity)

Both models preserve:
1. Fixup Dirac-Delta Warm-Start (Guaranteed >= Bicubic at step 0)
2. Occlusion-Gated Windowed SCAM (17 epipolar hypotheses, 100% Vectorized)
3. Compact Harmonic Coordinate Encoding (Continuous INR zoom)
4. Fast 4x Sub-Pixel Scanout for Real-Time VR execution
"""

import os
import sys
import time
import math
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
from scripts.training.train_quest_stereoinr_on_wild_data import StereoWildDataset, StereoVisionPerceptualLoss, calc_psnr


def train_single_tier(tier_name: str, width: int, num_blocks: int, epochs: int = 5, batch_size: int = 4, lr: float = 3e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 95)
    print(f">>> TRAINING MULTI-SCALE TIER: {tier_name.upper()} (Width={width}, Blocks={num_blocks})")
    print(f"[*] Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 95)

    save_dir = os.path.join(ROOT_DIR, 'weights', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    out_ckpt = os.path.join(save_dir, f'quest_stereoinr_{tier_name.lower()}_wild_best.pth')

    # Initialize model with Fixup Dirac Delta warm-start
    model = QuestStereoINR(in_channels=3, width=width, num_blocks=num_blocks, coord_dim=16, min_disp=-2, max_disp=14).to(device)
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Initialized {tier_name} model with {train_params:,d} trainable parameters.")

    criterion = StereoVisionPerceptualLoss(lambda_photo=1.0, lambda_iogd=0.1, lambda_lap=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    dataset = StereoWildDataset(root_dir=ROOT_DIR, patch_size=96, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"[*] Training dataset: {len(dataset)} crops ({len(dataloader)} batches/epoch)")

    # Validation reference
    val_dataset = StereoWildDataset(root_dir=ROOT_DIR, patch_size=128, is_train=False)
    val_l, val_r = val_dataset[0]
    val_l = val_l.unsqueeze(0).to(device)
    val_r = val_r.unsqueeze(0).to(device)
    val_lr_l = F.interpolate(val_l, scale_factor=0.25, mode='bicubic', align_corners=False)
    val_lr_r = F.interpolate(val_r, scale_factor=0.25, mode='bicubic', align_corners=False)

    bicubic_ref = F.interpolate(val_lr_l, scale_factor=4.0, mode='bicubic', align_corners=False)
    base_psnr = calc_psnr(bicubic_ref, val_l)
    print(f"[*] Validation Bicubic Baseline PSNR: {base_psnr:.2f} dB")

    best_psnr = base_psnr

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for batch_idx, (batch_hr_l, batch_hr_r) in enumerate(dataloader):
            batch_hr_l = batch_hr_l.to(device)
            batch_hr_r = batch_hr_r.to(device)

            # Continuous Arbitrary Scale: 60% 4.0x, 40% arbitrary [1.5, 3.8]
            if random.random() < 0.6:
                scale = 4.0
                force_cont = (random.random() < 0.3)
            else:
                scale = round(random.uniform(1.5, 3.8), 2)
                force_cont = True

            # Generate LR pair
            inv_scale = 1.0 / scale
            lr_l = F.interpolate(batch_hr_l, scale_factor=inv_scale, mode='bicubic', align_corners=False)
            lr_r = F.interpolate(batch_hr_r, scale_factor=inv_scale, mode='bicubic', align_corners=False)

            # Target size matching
            target_h = int(round(lr_l.shape[2] * scale))
            target_w = int(round(lr_l.shape[3] * scale))
            tgt_l = F.interpolate(batch_hr_l, size=(target_h, target_w), mode='bicubic', align_corners=False)
            tgt_r = F.interpolate(batch_hr_r, size=(target_h, target_w), mode='bicubic', align_corners=False)

            optimizer.zero_grad()
            pred_l, pred_r = model(lr_l, lr_r, scale=scale, force_continuous=force_cont)

            if pred_l.shape[2:] != tgt_l.shape[2:]:
                pred_l = F.interpolate(pred_l, size=tgt_l.shape[2:], mode='bilinear', align_corners=False)
                pred_r = F.interpolate(pred_r, size=tgt_r.shape[2:], mode='bilinear', align_corners=False)

            loss, l_ph, l_io, l_lap = criterion(pred_l, pred_r, tgt_l, tgt_r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - t0
        avg_loss = running_loss / max(len(dataloader), 1)

        # Validation
        model.eval()
        with torch.no_grad():
            sr_val_l, _ = model(val_lr_l, val_lr_r, scale=4.0, force_continuous=False)
            val_psnr = calc_psnr(sr_val_l, val_l)

        status = "NEW BEST (*)" if val_psnr > best_psnr else ""
        print(f"Epoch {epoch:2d}/{epochs:2d} ({epoch_time:4.1f}s) | Loss: {avg_loss:.4f} | Val PSNR: {val_psnr:5.2f} dB (Gain: {val_psnr - base_psnr:+5.2f} dB) {status}")

        if val_psnr >= best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'tier': tier_name,
                'width': width,
                'num_blocks': num_blocks,
                'model_state_dict': model.state_dict(),
                'val_psnr': val_psnr,
                'gain': val_psnr - base_psnr
            }, out_ckpt)

    print(f"[+] Successfully finished training {tier_name}! Saved checkpoint to: {out_ckpt} (Best PSNR: {best_psnr:.2f} dB)")
    return out_ckpt


if __name__ == '__main__':
    print(">>> LAUNCHING MULTI-SCALE ARCHITECTURAL TRAINING SUITE")
    # Train Medium (448K params)
    train_single_tier(tier_name='medium', width=80, num_blocks=8, epochs=4, batch_size=4, lr=4e-4)
    # Train Large (1.78M params)
    train_single_tier(tier_name='large', width=128, num_blocks=16, epochs=3, batch_size=2, lr=2.5e-4)
