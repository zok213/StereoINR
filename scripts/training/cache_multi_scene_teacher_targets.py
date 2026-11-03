"""
Precomputes Offline Knowledge Targets from Heavyweight Teachers:
- Teacher 1: HAT (CVPR 2023, 20.77M params) -> 5.3s per forward pass
- Teacher 2: NAFRSSR-M (CVPRW 2022, 280,383 params) -> 4.8ms per forward pass

Caches 20 diverse multi-dataset patches into datasets/distillation_cache/
Enables rapid, scalable student training in seconds without repeated teacher bottlenecks.
"""

import os
import sys
import time
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'external', 'NAFRSSR'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'external', 'HAT'))

from NAFRSSR_M_Net import NAFSSR
from hat.archs.hat_arch import HAT


def load_tensor(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def generate_synthetic_reading_patch(H: int = 192, W: int = 192, disparity: int = 6, device: torch.device = torch.device('cpu')):
    bg = torch.ones(1, 3, H, W, device=device) * 0.95
    for y in range(20, H - 20, 24):
        bg[:, :, y:y+3, 15:W-15] = 0.08
        for x in range(25, W - 25, 12):
            bg[:, :, y-10:y+3, x:x+2] = 0.08
            if (x // 12) % 2 == 0:
                bg[:, :, y-6:y-3, x:x+6] = 0.08

    hr_l = bg
    hr_r = torch.roll(hr_l, shifts=-disparity, dims=3)
    hr_r[:, :, :, -disparity:] = 0.95

    lr_l = F.interpolate(hr_l, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
    lr_r = F.interpolate(hr_r, scale_factor=0.25, mode='bicubic', align_corners=False).clamp(0, 1)
    return hr_l, hr_r, lr_l, lr_r


def precompute_distillation_cache():
    print("=" * 105)
    print(">>> PRECOMPUTING OFFLINE TEACHER TARGETS: HAT (20.77M) + NAFRSSR-M (280K)")
    print("=" * 105)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n")

    cache_dir = os.path.join(ROOT_DIR, 'datasets', 'distillation_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Load Teachers
    print("[*] Loading HAT (20.77M params)...")
    hat = HAT(
        upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3,
        squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.,
        depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
    ).to(device).eval()
    ckpt_hat = torch.load('external/HAT/weights/HAT_SRx4_ImageNet-pretrain.pth', map_location=device, weights_only=False)
    hat.load_state_dict(ckpt_hat['params_ema'] if 'params_ema' in ckpt_hat else ckpt_hat['params'], strict=True)

    print("[*] Loading NAFRSSR-M (280,383 params)...")
    naf = NAFSSR(up_scale=4, width=64, num_blks=16).to(device).eval()
    ckpt_naf = torch.load('external/NAFRSSR/NAFRSSR_M.pth', map_location=device, weights_only=False)
    naf.load_state_dict(ckpt_naf['params'] if 'params' in ckpt_naf else ckpt_naf, strict=True)

    # 2. Collect Scenes
    scene_crops = []
    patch_size = 48

    # A. Middlebury (5 scenes)
    mb_dir = os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Middlebury')
    for sc in ['cloth2', 'motorcycle', 'piano', 'pipes', 'sword2']:
        hl = os.path.join(mb_dir, 'hr', sc, 'hr0.png')
        hr = os.path.join(mb_dir, 'hr', sc, 'hr1.png')
        ll = os.path.join(mb_dir, 'lr_x4', sc, 'lr0.png')
        lr = os.path.join(mb_dir, 'lr_x4', sc, 'lr1.png')
        if os.path.exists(hl) and os.path.exists(ll):
            t_hl, t_hr = load_tensor(hl, device), load_tensor(hr, device)
            t_ll, t_lr = load_tensor(ll, device), load_tensor(lr, device)
            # central crop
            cy = (t_ll.shape[2] - patch_size) // 2
            cx = (t_ll.shape[3] - patch_size) // 2
            scene_crops.append((
                f"Middlebury_{sc}",
                t_ll[:, :, cy:cy+patch_size, cx:cx+patch_size],
                t_lr[:, :, cy:cy+patch_size, cx:cx+patch_size],
                t_hl[:, :, cy*4:(cy+patch_size)*4, cx*4:(cx+patch_size)*4],
                t_hr[:, :, cy*4:(cy+patch_size)*4, cx*4:(cx+patch_size)*4],
            ))

    # B. Flickr1024 (6 diverse scenes)
    fl_dir = os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'Flickr1024')
    for sc in sorted(os.listdir(os.path.join(fl_dir, 'hr')))[:6]:
        hl = os.path.join(fl_dir, 'hr', sc, 'hr0.png')
        hr = os.path.join(fl_dir, 'hr', sc, 'hr1.png')
        ll = os.path.join(fl_dir, 'lr_x4', sc, 'lr0.png')
        lr = os.path.join(fl_dir, 'lr_x4', sc, 'lr1.png')
        if os.path.exists(hl) and os.path.exists(ll):
            t_hl, t_hr = load_tensor(hl, device), load_tensor(hr, device)
            t_ll, t_lr = load_tensor(ll, device), load_tensor(lr, device)
            cy = (t_ll.shape[2] - patch_size) // 2
            cx = (t_ll.shape[3] - patch_size) // 2
            scene_crops.append((
                f"Flickr1024_{sc}",
                t_ll[:, :, cy:cy+patch_size, cx:cx+patch_size],
                t_lr[:, :, cy:cy+patch_size, cx:cx+patch_size],
                t_hl[:, :, cy*4:(cy+patch_size)*4, cx*4:(cx+patch_size)*4],
                t_hr[:, :, cy*4:(cy+patch_size)*4, cx*4:(cx+patch_size)*4],
            ))

    # C. KITTI2015 (3 scenes)
    kt_dir = os.path.join(ROOT_DIR, 'datasets', 'StereoSR', 'test', 'KITTI2015')
    for sc in sorted(os.listdir(os.path.join(kt_dir, 'hr')))[:3]:
        hl = os.path.join(kt_dir, 'hr', sc, 'hr0.png')
        hr = os.path.join(kt_dir, 'hr', sc, 'hr1.png')
        ll = os.path.join(kt_dir, 'lr_x4', sc, 'lr0.png')
        lr = os.path.join(kt_dir, 'lr_x4', sc, 'lr1.png')
        if os.path.exists(hl) and os.path.exists(ll):
            t_hl, t_hr = load_tensor(hl, device), load_tensor(hr, device)
            t_ll, t_lr = load_tensor(ll, device), load_tensor(lr, device)
            cy = (t_ll.shape[2] - patch_size) // 2
            cx = (t_ll.shape[3] - patch_size) // 2
            scene_crops.append((
                f"KITTI2015_{sc}",
                t_ll[:, :, cy:cy+patch_size, cx:cx+patch_size],
                t_lr[:, :, cy:cy+patch_size, cx:cx+patch_size],
                t_hl[:, :, cy*4:(cy+patch_size)*4, cx*4:(cx+patch_size)*4],
                t_hr[:, :, cy*4:(cy+patch_size)*4, cx*4:(cx+patch_size)*4],
            ))

    # D. Synthetic Phone Reading Text (2 variants)
    for disp in [4, 8]:
        s_hl, s_hr, s_ll, s_lr = generate_synthetic_reading_patch(192, 192, disparity=disp, device=device)
        scene_crops.append((
            f"PhoneText_disp{disp}",
            s_ll[:, :, :patch_size, :patch_size],
            s_lr[:, :, :patch_size, :patch_size],
            s_hl[:, :, :patch_size*4, :patch_size*4],
            s_hr[:, :, :patch_size*4, :patch_size*4],
        ))

    print(f"[*] Processing {len(scene_crops)} High-Value Patches...")
    t0 = time.time()

    for idx, (sname, p_lr_l, p_lr_r, p_hr_l, p_hr_r) in enumerate(scene_crops):
        t_p0 = time.time()
        with torch.no_grad():
            # HAT inference
            hat_out_l = hat(p_lr_l).clamp(0, 1).cpu()

            # NAFR-SSR inference
            inp_naf = torch.cat([p_lr_l, p_lr_r], dim=1)
            naf_out = naf(inp_naf).clamp(0, 1)
            naf_out_l = naf_out[:, 0:3, :, :].cpu()
            naf_out_r = naf_out[:, 3:6, :, :].cpu()

        cache_data = {
            'name': sname,
            'lr_l': p_lr_l.cpu(),
            'lr_r': p_lr_r.cpu(),
            'hr_l': p_hr_l.cpu(),
            'hr_r': p_hr_r.cpu(),
            'hat_target_l': hat_out_l,
            'naf_target_l': naf_out_l,
            'naf_target_r': naf_out_r,
        }

        save_file = os.path.join(cache_dir, f"patch_{idx:03d}_{sname}.pt")
        torch.save(cache_data, save_file)
        dt = time.time() - t_p0
        print(f"[{idx+1:02d}/{len(scene_crops):02d}] Cached {sname:20s} -> {save_file} ({dt:.2f}s)")

    print(f"\n[OK] Successfully cached all {len(scene_crops)} multi-dataset teacher targets in {time.time() - t0:.1f}s!")


if __name__ == '__main__':
    precompute_distillation_cache()
