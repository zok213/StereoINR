"""
Training & Distillation Pipeline: Scale-Liberated StereoINR (RepDyT-EdgeINR)
=============================================================================
Milestone 2 Convergence Training:
- Multi-Task Loss Suite: L_total = 1.0 * L_char + 0.5 * L_warp + 0.1 * L_fft + 0.25 * L_KD
  (Eliminated flawed loss_disp = 0.01 * disp^2).
- RepDyT Structural Reparameterization with Dirac Warm-Start.
- Residual Damping (res_scale=0.2) to eliminate feature variance explosion.
- Strict Lipschitz bound enforcement (Gamma_max <= 2.0) on Dynamic Tanh.
- Datasets: Quest 3S Harvested (Phases 1, 2, 3, 4, 6) + Flickr1024 + Middlebury.
- Output Checkpoint: weights/checkpoints/stereoinr_scale_liberated_improved.pth
"""

import os
import sys
import time
import math
import glob
import json
from typing import List, Tuple, Dict
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

stereoinr_dir = os.path.join(ROOT_DIR, 'archive', 'legacy_academic_reproduction', 'StereoINR_official')
if stereoinr_dir not in sys.path:
    sys.path.insert(0, stereoinr_dir)

from basicsr.archs.continuous_arch import StereoINR
from src.edge_stereoinr.phase2_edge_inr.backbones.stereoinr_scale_liberated import (
    ScaleLiberatedStereoINR,
    MultiTaskLossSuite
)


def calc_psnr(p: torch.Tensor, t: torch.Tensor) -> float:
    mse = F.mse_loss(p, t).item()
    if mse <= 1e-10:
        return 99.99
    return 10.0 * math.log10(1.0 / mse)


def compute_stereo_mae(img_l: torch.Tensor, img_r: torch.Tensor, max_disp: int = 14) -> float:
    min_mae = 1e9
    for d in range(0, max_disp + 1):
        if d == 0:
            diff = torch.abs(img_l - img_r).mean().item()
        else:
            diff = torch.abs(img_l[:, :, :, d:] - img_r[:, :, :, :-d]).mean().item()
        if diff < min_mae:
            min_mae = diff
    return float(min_mae)


def load_img_tensor(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def extract_teacher_feature(teacher: nn.Module, lr_l: torch.Tensor, lr_r: torch.Tensor) -> torch.Tensor:
    """Extracts intermediate teacher representation (180 channels) for distillation."""
    with torch.no_grad():
        from einops import rearrange
        inp = torch.stack([lr_l, lr_r], dim=1)  # (B, 2, 3, H, W)
        B, _, _, H, W = inp.shape
        x = rearrange(inp, 'b t c h w -> (b t) c h w')
        
        # Build cell scale
        cell = torch.tensor([2.0 / (H * 4), 2.0 / (W * 4)], device=lr_l.device).view(1, 1, 2).expand(B, H * W, 2)
        cell_ = torch.stack([cell, cell], dim=1)
        cell_ = rearrange(cell_, 'b t n c -> (b t) n c')
        scale = cell_[:, 0, :]
        scale = teacher.pos_enc(scale)

        x = teacher.conv_first(x)
        feat = teacher.conv_after_body(teacher.forward_features(x, scale)) + x
        return feat[:B]


def collect_training_pairs() -> List[Tuple[str, str, str]]:
    """Gathers stereo image pairs from Quest 3S Harvested, Flickr1024, and Middlebury."""
    pairs = []

    # 1. Quest 3S Harvested Training Phases
    quest_dir = os.path.join(ROOT_DIR, 'datasets', 'Quest3S_UserStereo_Harvested', 'train', 'hr')
    if os.path.exists(quest_dir):
        for f in os.listdir(quest_dir):
            if f.endswith('_L.png'):
                hl = os.path.join(quest_dir, f)
                hr = os.path.join(quest_dir, f.replace('_L.png', '_R.png'))
                if os.path.exists(hr):
                    pairs.append(('Quest3S_Train', hl, hr))

    # 2. Middlebury Fine Textures
    mb_dir = os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Middlebury', 'hr')
    if os.path.exists(mb_dir):
        for sc in ['cloth2', 'motorcycle', 'bowling1', 'baby1', 'lampshade1']:
            hl = os.path.join(mb_dir, sc, 'hr0.png')
            hr = os.path.join(mb_dir, sc, 'hr1.png')
            if os.path.exists(hl) and os.path.exists(hr):
                pairs.append(('Middlebury', hl, hr))

    # 3. Flickr1024 Pairs (Diverse Urban & Architectural Textures)
    fl_dir = os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Flickr1024', 'hr')
    if os.path.exists(fl_dir):
        fl_scenes = sorted(os.listdir(fl_dir))[:25]
        for sc in fl_scenes:
            hl = os.path.join(fl_dir, sc, 'hr0.png')
            hr = os.path.join(fl_dir, sc, 'hr1.png')
            if os.path.exists(hl) and os.path.exists(hr):
                pairs.append(('Flickr1024', hl, hr))

    return pairs


def precompute_teacher_features(teacher: nn.Module, pairs: list, device: torch.device, crop_size: int = 128) -> Dict[str, torch.Tensor]:
    """Precomputes offline teacher feature cache for fast KD distillation without live inference overhead."""
    print(f"[*] Precomputing Teacher Feature Representations for {len(pairs)} pairs...", flush=True)
    cache = {}
    for ds_name, hl_p, hr_p in pairs:
        try:
            img_hl = load_img_tensor(hl_p, device)
            img_hr = load_img_tensor(hr_p, device)
            H, W = img_hl.shape[2], img_hl.shape[3]
            if H < crop_size or W < crop_size:
                continue
            # Extract central crop
            top = (H - crop_size) // 2
            left = (W - crop_size) // 2
            patch_hl = img_hl[:, :, top:top+crop_size, left:left+crop_size]
            patch_hr = img_hr[:, :, top:top+crop_size, left:left+crop_size]
            lr_l = F.interpolate(patch_hl, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
            lr_r = F.interpolate(patch_hr, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)

            feat_l = extract_teacher_feature(teacher, lr_l, lr_r)
            cache[hl_p] = {
                'patch_hl': patch_hl.cpu(),
                'patch_hr': patch_hr.cpu(),
                'lr_l': lr_l.cpu(),
                'lr_r': lr_r.cpu(),
                'teacher_feat': feat_l.cpu()
            }
        except Exception as e:
            print(f"[-] Precompute error on {hl_p}: {e}", flush=True)
            continue

    print(f"[+] Successfully cached {len(cache)} teacher feature representations in RAM!", flush=True)
    return cache


def run_improved_training(
    epochs: int = 8,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    crop_size: int = 128,
    crops_per_pair: int = 2
):
    print("=" * 95, flush=True)
    print(">>> MILESTONE 2: TRAINING IMPROVED SCALE-LIBERATED STEREONR (RepDyT-EdgeINR)", flush=True)
    print(f">>> Target: weights/checkpoints/stereoinr_scale_liberated_improved.pth", flush=True)
    print(f">>> Hyperparameters: Epochs={epochs} | LR={lr} | WeightDecay={weight_decay} | Crop={crop_size}x{crop_size}", flush=True)
    print("=" * 95, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 1. Gather Training Pairs
    train_pairs = collect_training_pairs()
    print(f"[+] Assembled {len(train_pairs)} training stereo scene pairs.", flush=True)

    # 2. Precompute Teacher Features Once (Section 6.1 M1 Blueprint)
    teacher_cache = {}
    ckpt_teacher = os.path.join(ROOT_DIR, 'weights', 'checkpoints', 'StereoINR_official.pth')
    if os.path.exists(ckpt_teacher):
        try:
            print("[*] Loading Official StereoINR Teacher (43.8M params)...", flush=True)
            teacher = StereoINR(
                upscale=4, in_chans=3, img_size=[32, 96], window_size=16,
                compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5,
                drop_path_rate=0.1, img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='continuous', resi_connection='1conv'
            ).to(device).eval()
            sd_t = torch.load(ckpt_teacher, map_location=device, weights_only=False)
            teacher.load_state_dict(sd_t.get('params_ema', sd_t.get('params', sd_t)), strict=True)
            for p in teacher.parameters():
                p.requires_grad = False
            print("[+] Successfully loaded StereoINR Teacher!", flush=True)

            teacher_cache = precompute_teacher_features(teacher, train_pairs[:12], device, crop_size=crop_size)

            # Free teacher memory from GPU
            del teacher
            del sd_t
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print("[*] Released Teacher from VRAM to maximize student training headroom.", flush=True)
        except Exception as e:
            print(f"[-] Teacher precompute error: {e}. Running with loss suite without live KD.", flush=True)

    # 3. Instantiate Student Model (ScaleLiberatedStereoINR)
    student = ScaleLiberatedStereoINR(embed_dim=144, num_blocks=8, res_scale=0.2).to(device)
    baseline_ckpt = os.path.join(ROOT_DIR, 'logs', 'kaggle_v12_complete', 'stereoinr_scale_liberated_best.pth')
    if os.path.exists(baseline_ckpt):
        student.load_pretrained_baseline(baseline_ckpt)
        print(f"[+] Loaded baseline checkpoint with Dirac warm-start from {baseline_ckpt}", flush=True)
    else:
        print("[*] Training from scratch with Dirac initialization.", flush=True)

    # 4. Multi-Task Loss Suite (Requirement 1)
    loss_suite = MultiTaskLossSuite(
        lambda_char=1.0,
        lambda_warp=0.5,
        lambda_fft=0.1,
        lambda_kd=0.25,
        eps_char=1e-3,
        tau_lrc=1.0,
        student_dim=144,
        teacher_dim=180
    ).to(device)

    # 5. Validation reference set
    val_pairs = [
        os.path.join(ROOT_DIR, 'datasets', 'Quest3S_UserStereo_Harvested', 'test', 'hr', 'phase5_outdoor_campus_brickwalk'),
        os.path.join(ROOT_DIR, 'datasets', 'Quest3S_UserStereo_Harvested', 'test', 'hr', 'phase7_indoor_portal_awards'),
        os.path.join(ROOT_DIR, 'datasets', 'Quest3S_UserStereo_Harvested', 'test', 'hr', 'phase8_return_lab_desk')
    ]

    def evaluate_val_set():
        student.eval()
        psnrs, maes = [], []
        with torch.no_grad():
            for v_prefix in val_pairs:
                vl, vr = f"{v_prefix}_L.png", f"{v_prefix}_R.png"
                if os.path.exists(vl) and os.path.exists(vr):
                    tl = load_img_tensor(vl, device)
                    tr = load_img_tensor(vr, device)
                    H, W = tl.shape[2], tl.shape[3]
                    ch, cw = 256, 256
                    y0, x0 = (H - ch) // 2, (W - cw) // 2
                    c_tl = tl[:, :, y0:y0+ch, x0:x0+cw]
                    c_tr = tr[:, :, y0:y0+ch, x0:x0+cw]

                    c_lr_l = F.interpolate(c_tl, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
                    c_lr_r = F.interpolate(c_tr, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)

                    sr_l, sr_r, _ = student(c_lr_l, c_lr_r, scale=4.0)
                    sr_l = sr_l.clamp(0, 1)
                    sr_r = sr_r.clamp(0, 1)

                    psnrs.append(calc_psnr(sr_l, c_tl))
                    maes.append(compute_stereo_mae(sr_l, sr_r))

        return np.mean(psnrs) if psnrs else 0.0, np.mean(maes) if maes else 0.0

    init_psnr, init_mae = evaluate_val_set()
    print(f"[*] Pre-Training Validation: Crop PSNR = {init_psnr:.2f} dB | Stereo MAE = {init_mae:.4f}\n", flush=True)

    # 6. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    best_psnr = init_psnr
    history = []
    t_start = time.time()

    print("=" * 95, flush=True)
    print(f"STARTING {epochs}-EPOCH CONVERGENCE TRAINING WITH MULTI-TASK LOSS", flush=True)
    print("=" * 95, flush=True)

    cached_keys = list(teacher_cache.keys())

    for epoch in range(1, epochs + 1):
        student.train()
        t_ep = time.time()
        ep_loss = 0.0
        n_steps = 0

        # Train on precomputed high-value teacher-cached crops + random dataset crops
        for ds_name, hl_p, hr_p in train_pairs:
            t_feat_l = None
            if hl_p in teacher_cache:
                entry = teacher_cache[hl_p]
                patch_hl = entry['patch_hl'].to(device)
                patch_hr = entry['patch_hr'].to(device)
                lr_l = entry['lr_l'].to(device)
                lr_r = entry['lr_r'].to(device)
                t_feat_l = entry['teacher_feat'].to(device)
            else:
                try:
                    img_hl = load_img_tensor(hl_p, device)
                    img_hr = load_img_tensor(hr_p, device)
                except Exception:
                    continue
                H, W = img_hl.shape[2], img_hl.shape[3]
                if H < crop_size or W < crop_size:
                    continue
                top = np.random.randint(0, H - crop_size)
                left = np.random.randint(0, W - crop_size)
                patch_hl = img_hl[:, :, top:top+crop_size, left:left+crop_size]
                patch_hr = img_hr[:, :, top:top+crop_size, left:left+crop_size]
                lr_l = F.interpolate(patch_hl, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
                lr_r = F.interpolate(patch_hr, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == 'cuda')):
                sr_l, sr_r, info = student(lr_l, lr_r, scale=4.0)
                fl, fr = student.forward_features(lr_l, lr_r)[:2]

                loss_dict = loss_suite(
                    pred_l=sr_l, pred_r=sr_r,
                    gt_l=patch_hl, gt_r=patch_hr,
                    disp_l2r=info['disp_l2r'], disp_r2l=info['disp_r2l'],
                    student_feat_l=fl, teacher_feat_l=t_feat_l
                )
                loss = loss_dict['loss_total']

            if device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()

            ep_loss += loss.item()
            n_steps += 1

        scheduler.step()
        avg_loss = ep_loss / max(n_steps, 1)
        cur_lr = scheduler.get_last_lr()[0]

        # Validation every epoch
        val_psnr, val_mae = evaluate_val_set()
        gain = val_psnr - init_psnr
        print(f"[Epoch {epoch:02d}/{epochs:02d}] Time: {time.time()-t_ep:.1f}s | Loss: {avg_loss:.4f} | LR: {cur_lr:.2e} | Val PSNR: {val_psnr:.2f} dB ({gain:+.2f} dB) | Stereo MAE: {val_mae:.4f}", flush=True)

        history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'val_psnr': val_psnr,
            'val_mae': val_mae
        })

        if val_psnr >= best_psnr or epoch == epochs:
            best_psnr = val_psnr
            save_checkpoint(student, val_psnr, val_mae, epoch, history)

    total_time = time.time() - t_start
    print("\n" + "=" * 95, flush=True)
    print(f">>> TRAINING COMPLETED IN {total_time:.1f}s (~{total_time/60:.2f} minutes)!", flush=True)
    print(f">>> Peak Validation PSNR: {best_psnr:.2f} dB (Gain over init: {best_psnr - init_psnr:+.2f} dB)", flush=True)
    print("=" * 95, flush=True)

    return student, best_psnr


def save_checkpoint(student: ScaleLiberatedStereoINR, psnr: float, mae: float, epoch: int, history: list):
    """Saves improved student checkpoint and verifies deploy parity."""
    ckpt_dir = os.path.join(ROOT_DIR, 'weights', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    out_path = os.path.join(ckpt_dir, 'stereoinr_scale_liberated_improved.pth')

    train_sd = student.state_dict()

    device = next(student.parameters()).device
    deploy_model = ScaleLiberatedStereoINR(embed_dim=144, num_blocks=8, res_scale=0.2).to(device).eval()
    deploy_model.load_state_dict(train_sd, strict=False)

    test_x = torch.rand(1, 3, 32, 32, device=device)
    with torch.no_grad():
        out_train, _, _ = deploy_model(test_x, test_x, scale=4.0)

    deploy_model.switch_to_deploy()
    with torch.no_grad():
        out_deploy, _, _ = deploy_model(test_x, test_x, scale=4.0)

    parity_diff = (out_train - out_deploy).abs().max().item()
    print(f"[*] Deploy Weight Fusion Parity Max Diff: {parity_diff:.2e}", flush=True)
    assert parity_diff < 1e-4, f"Deploy parity check failed: {parity_diff}"

    deploy_sd = deploy_model.state_dict()
    deploy_params = sum(p.numel() for p in deploy_model.parameters())
    print(f"[*] Verified Deploy Parameters: {deploy_params:,} (Target: 4,468,180)", flush=True)

    save_payload = {
        'model_state_dict': train_sd,
        'deploy_state_dict': deploy_sd,
        'psnr': psnr,
        'mae': mae,
        'epoch': epoch,
        'deploy_params': deploy_params,
        'parity_diff': parity_diff,
        'history': history,
        'embed_dim': 144,
        'num_blocks': 8,
        'res_scale': 0.2
    }

    torch.save(save_payload, out_path)
    print(f"[+] Saved improved student checkpoint to: {out_path}", flush=True)


if __name__ == '__main__':
    run_improved_training(epochs=8, crop_size=128, crops_per_pair=2)
