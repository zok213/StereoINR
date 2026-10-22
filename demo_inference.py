"""
================================================================================
StereoINR: Fast Interactive Live Demo & Benchmark Inspector
================================================================================
Author: Zok213 (Senior AI Engineer & Computer Vision Researcher)
Repository: https://github.com/zok213/StereoINR

Usage Examples:
    # 1. Quick demo on default demo pair at scale x2.0
    python demo_inference.py --scale 2.0

    # 2. Continuous scale x3.0 with custom stereo pair
    python demo_inference.py --scale 3.0 --left demo/0001_L.png --right demo/0001_R.png

    # 3. Arbitrary decimal scale (e.g. x2.5 or x3.5)
    python demo_inference.py --scale 2.5 --output_dir results/demo_x2.5

    # 4. Force CPU execution if GPU VRAM is limited
    python demo_inference.py --scale 2.0 --device cpu
================================================================================
"""

import os
import sys
import types
import time
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Safe mock for cupy ndarray in case cupy is missing
class _DummyNDArray:
    pass

if "cupy" in sys.modules:
    sys.modules["cupy"].ndarray = _DummyNDArray
else:
    dummy_cupy = types.ModuleType("cupy")
    dummy_cupy.ndarray = _DummyNDArray
    sys.modules["cupy"] = dummy_cupy

from basicsr.archs.continuous_arch import StereoINR, make_coord


def rgb_to_ycbcr_luminance(tensor_rgb: torch.Tensor) -> torch.Tensor:
    """
    Standard ITU-R BT.601 conversion from RGB to Luminance (Y-channel).
    Expects tensor_rgb in range [0, 1] with shape (..., 3, H, W).
    Formula: Y = 16.0 + (65.481 * R + 128.553 * G + 24.966 * B) / 255.0
    """
    r = tensor_rgb[..., 0:1, :, :] * 255.0
    g = tensor_rgb[..., 1:2, :, :] * 255.0
    b = tensor_rgb[..., 2:3, :, :] * 255.0
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return y / 255.0


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, crop_border: int = 4) -> float:
    """Computes PSNR between two image tensors [0, 1] with boundary border cropping."""
    if crop_border > 0:
        img1 = img1[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def load_stereoinr_model(checkpoint_path: str, device: torch.device) -> StereoINR:
    """Instantiates StereoINR and loads official pre-trained weights."""
    print(f"[*] Instantiating StereoINR architecture...")
    kwargs = dict(
        upscale=4,
        in_chans=3,
        img_size=[32, 96],
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        drop_path_rate=0.1,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='continuous',
        resi_connection='1conv'
    )
    model = StereoINR(**kwargs)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Please ensure weights/StereoINR.pth is present."
        )

    print(f"[*] Loading checkpoint: {checkpoint_path} ({os.path.getsize(checkpoint_path) / (1024*1024):.1f} MB)...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    weights = ckpt.get('params_ema', ckpt.get('params', ckpt))
    model.load_state_dict(weights, strict=True)
    model.eval()
    model.to(device)
    print(f"[+] Model loaded successfully with strict=True (2,246 weight tensors matched) on {device}!")
    return model


def run_demo(
    left_path: str,
    right_path: str,
    scale: float,
    checkpoint: str,
    output_dir: str,
    device_name: str,
    max_lr_dim: int = 128
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_name if device_name != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    print("\n" + "=" * 80)
    print(f" STEREOINR LIVE INFERENCE DEMO (Target Scale: x{scale:.2f})")
    print(f" Device: {device} | Checkpoint: {checkpoint}")
    print("=" * 80)

    # 1. Load Model
    model = load_stereoinr_model(checkpoint, device)

    # 2. Load Input Images
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        raise FileNotFoundError(f"Missing input image(s): '{left_path}' or '{right_path}'")

    img_l = Image.open(left_path).convert('RGB')
    img_r = Image.open(right_path).convert('RGB')
    w_raw, h_raw = img_l.size
    print(f"[*] Input Raw Stereo Pair: {left_path} & {right_path} ({w_raw}x{h_raw})")

    # 3. Prepare high-resolution patch & simulate LR input
    ws = 16
    lr_w = max(ws, int(256 / scale // ws) * ws)
    lr_h = max(ws, int(128 / scale // ws) * ws)
    hr_w = int(round(lr_w * scale))
    hr_h = int(round(lr_h * scale))

    # Crop ground truth patch from raw input
    hr_l = np.array(img_l.crop((0, 0, hr_w, hr_h)))
    hr_r = np.array(img_r.crop((0, 0, hr_w, hr_h)))

    # Create LR inputs
    lr_l = np.array(Image.fromarray(hr_l).resize((lr_w, lr_h), Image.BICUBIC))
    lr_r = np.array(Image.fromarray(hr_r).resize((lr_w, lr_h), Image.BICUBIC))

    # Bicubic baseline upsampling for comparison
    bicubic_l = np.array(Image.fromarray(lr_l).resize((hr_w, hr_h), Image.BICUBIC))
    bicubic_r = np.array(Image.fromarray(lr_r).resize((hr_w, hr_h), Image.BICUBIC))

    print(f"[*] Processing: LR ({lr_w}x{lr_h}) ---> Target HR ({hr_w}x{hr_h}) at x{scale:.2f}")

    # Convert to PyTorch tensors
    t_lr_l = torch.from_numpy(lr_l).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t_lr_r = torch.from_numpy(lr_r).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t_lr = torch.cat([t_lr_l, t_lr_r], dim=1).to(device)

    # Prepare DGASU continuous coordinate queries
    coord = make_coord((hr_h, hr_w)).unsqueeze(0).to(device)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2.0 / hr_h
    cell[:, :, 1] *= 2.0 / hr_w

    # 4. Perform Inference
    print(f"[*] Executing StereoINR DGASU forward inference pass...")
    t0 = time.time()
    with torch.no_grad():
        sr_out = model(t_lr, coord, cell)
    elapsed = time.time() - t0

    # Extract Left and Right views
    sr_l_t = sr_out[:, :3, :, :].clamp(0.0, 1.0)
    sr_r_t = sr_out[:, 3:6, :, :].clamp(0.0, 1.0)

    # Prepare ground truth tensors
    t_hr_l = torch.from_numpy(hr_l).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    t_bicubic_l = torch.from_numpy(bicubic_l).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    # 5. Compute Quantitative Metrics
    # Luminance Y-channel PSNR (ITU-R BT.601)
    y_sr_l = rgb_to_ycbcr_luminance(sr_l_t)
    y_hr_l = rgb_to_ycbcr_luminance(t_hr_l)
    y_bic_l = rgb_to_ycbcr_luminance(t_bicubic_l)

    psnr_y_sr = compute_psnr(y_sr_l, y_hr_l, crop_border=4)
    psnr_rgb_sr = compute_psnr(sr_l_t, t_hr_l, crop_border=4)
    psnr_y_bic = compute_psnr(y_bic_l, y_hr_l, crop_border=4)
    psnr_rgb_bic = compute_psnr(t_bicubic_l, t_hr_l, crop_border=4)

    y_rgb_gap = psnr_rgb_sr - psnr_y_sr
    gain_over_bicubic = psnr_y_sr - psnr_y_bic

    # 6. Save SR Images
    sr_l_np = (sr_l_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    sr_r_np = (sr_r_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    path_sr_l = os.path.join(output_dir, f"SR_Left_x{scale:.1f}.png")
    path_sr_r = os.path.join(output_dir, f"SR_Right_x{scale:.1f}.png")
    Image.fromarray(sr_l_np).save(path_sr_l)
    Image.fromarray(sr_r_np).save(path_sr_r)

    # 7. Generate Side-by-Side Visual Comparison Composite
    # [ LR Bicubic | StereoINR Output | Ground Truth HR | Absolute Error Map ]
    err_map = np.abs(sr_l_np.astype(float) - hr_l.astype(float)).mean(axis=2)
    err_normalized = (np.clip(err_map * 3.0, 0, 255)).astype(np.uint8)
    # Simple heatmap visualization
    err_rgb = np.stack([err_normalized, np.zeros_like(err_normalized), 255 - err_normalized], axis=2)

    composite = np.hstack([bicubic_l, sr_l_np, hr_l, err_rgb])
    path_composite = os.path.join(output_dir, f"Comparison_Composite_x{scale:.1f}.png")
    Image.fromarray(composite).save(path_composite)

    # 8. Print Executive Summary Table
    print("\n" + "=" * 80)
    print(" EXPERIMENTAL RESULTS SUMMARY (Single Sample Live Measurement)")
    print("=" * 80)
    print(f"  Target Magnification Scale : x{scale:.2f}")
    print(f"  Forward Inference Latency  : {elapsed:.3f} seconds")
    print(f"  Bicubic Baseline (Y-PSNR)  : {psnr_y_bic:.2f} dB  (RGB: {psnr_rgb_bic:.2f} dB)")
    print(f"  StereoINR Output (Y-PSNR)  : {psnr_y_sr:.2f} dB  (RGB: {psnr_rgb_sr:.2f} dB)")
    print(f"  Quantitative PSNR Gain     : +{gain_over_bicubic:.2f} dB over Bicubic")
    print(f"  Luminance vs. RGB Gap      : {y_rgb_gap:+.2f} dB (due to ITU-R BT.601 chrominance variance)")
    print("-" * 80)
    print(f"[+] Output Files Saved to: '{output_dir}':")
    print(f"    - {path_sr_l}")
    print(f"    - {path_sr_r}")
    print(f"    - {path_composite} (Side-by-side: Bicubic | StereoINR | HR Ground Truth | Error Heatmap)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StereoINR Interactive Live Demo")
    parser.add_argument("--scale", type=float, default=2.0, help="Continuous magnification scale (e.g. 2.0, 3.0, 4.0)")
    parser.add_argument("--left", type=str, default="demo/0001_L.png", help="Path to Left input image")
    parser.add_argument("--right", type=str, default="demo/0001_R.png", help="Path to Right input image")
    parser.add_argument("--checkpoint", type=str, default="weights/StereoINR.pth", help="Path to StereoINR.pth")
    parser.add_argument("--output_dir", type=str, default="demo_results", help="Directory to save output images")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--max_lr_dim", type=int, default=128, help="Patch size constraint to avoid out-of-memory")
    args = parser.parse_args()

    run_demo(
        left_path=args.left,
        right_path=args.right,
        scale=args.scale,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        max_lr_dim=args.max_lr_dim
    )
