"""
StereoINR Feature Map Extractor and Stage-by-Stage Visualizer
Extracts intermediate feature representations across all key blocks and stages:
  1. Shallow Convolutional Feature Extraction (conv_first: 3 -> 180 Channels)
  2. Deep Residual Hybrid Attention Groups (RHAG):
     - Stage 1 (Layer 0) - Low-level structural boundaries & edge maps
     - Stage 4 (Layer 3) - Mid-level localized texture representations
     - Stage 8 (Layer 7) - High-level contextual semantic patterns
     - Stage 12 (Layer 11) - Deep stereo-correlated representations
  3. Global Residual Fusion (conv_after_body)
  4. Upsampler Conditioning (conv_before_upsample)
  5. DGASU Disparity & Cross-View Implicit Representation

Visualizations:
  - Channel Mean Activation Heatmap
  - Salient Max Channel Activation (Edge & boundary sharpness)
  - PCA 3-Component RGB Projection (Disentanglement visualization)
"""

import os
import sys
import types
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# Mock cupy to ensure compatibility on any environment
if "cupy" not in sys.modules:
    m = types.ModuleType("cupy")
    class _DummyNDArray: pass
    m.ndarray = _DummyNDArray
    sys.modules["cupy"] = m
elif not hasattr(sys.modules["cupy"], "ndarray"):
    class _DummyNDArray: pass
    sys.modules["cupy"].ndarray = _DummyNDArray

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from basicsr.archs.continuous_arch import StereoINR, make_coord


def colormap_viridis(val: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch viridis colormap approximation for [0, 1] input.
    Returns RGB tensor of shape (H, W, 3) in uint8 [0, 255].
    """
    val = val.clamp(0.0, 1.0)
    # Polynomial coefficients for viridis approximation
    r = 0.282 - 0.961 * val + 2.87 * val**2 - 1.25 * val**3
    g = 0.003 + 1.41 * val - 0.44 * val**2
    b = 0.33 + 1.54 * val - 4.12 * val**2 + 3.19 * val**3
    rgb = torch.stack([r, g, b], dim=-1).clamp(0.0, 1.0)
    return (rgb * 255.0).byte()


def colormap_inferno(val: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch inferno colormap approximation for [0, 1] input.
    Returns RGB tensor of shape (H, W, 3) in uint8 [0, 255].
    """
    val = val.clamp(0.0, 1.0)
    r = 0.0 + 1.5 * val + 0.3 * val**2
    g = 0.0 + 0.2 * val + 1.4 * val**3
    b = 0.05 + 1.2 * val - 1.8 * val**2 + 1.1 * val**3
    rgb = torch.stack([r, g, b], dim=-1).clamp(0.0, 1.0)
    return (rgb * 255.0).byte()


class FeatureExtractorHook:
    """PyTorch hook to extract intermediate activations."""
    def __init__(self):
        self.features = {}

    def hook_fn(self, name):
        def _hook(module, input, output):
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output
            if isinstance(out_tensor, torch.Tensor):
                self.features[name] = out_tensor.detach().cpu()
        return _hook


def pca_projection(feat_map, n_components=3):
    """
    Project C-dimensional feature map (C, H, W) into 3 RGB channels using SVD/PCA.
    Shows semantic feature clustering and textural disentanglement.
    """
    C, H, W = feat_map.shape
    X = feat_map.reshape(C, -1).T  # (H*W, C)
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean
    try:
        U, S, V = torch.pca_lowrank(X_centered, q=n_components)
        X_pca = torch.matmul(X_centered, V[:, :n_components])  # (H*W, 3)
        for i in range(n_components):
            c_min = X_pca[:, i].min()
            c_max = X_pca[:, i].max()
            if c_max > c_min:
                X_pca[:, i] = (X_pca[:, i] - c_min) / (c_max - c_min)
            else:
                X_pca[:, i] = 0.5
        rgb = X_pca.reshape(H, W, n_components)
        return (rgb.clamp(0.0, 1.0) * 255.0).byte()
    except Exception:
        # Fallback slice top 3 channels
        rgb = feat_map[:3].permute(1, 2, 0)
        c_min, c_max = rgb.min(), rgb.max()
        rgb = (rgb - c_min) / (c_max - c_min + 1e-8)
        return (rgb.clamp(0.0, 1.0) * 255.0).byte()


def create_annotated_panel(img_tensor, title_text, width=320, height=180):
    """Creates a captioned image panel with clean PIL rendering."""
    # Resize image to panel display size
    pil_img = Image.fromarray(img_tensor.numpy(), mode="RGB")
    pil_img = pil_img.resize((width, height - 30), Image.NEAREST)

    # Panel with top title header
    panel = Image.new("RGB", (width, height), color=(20, 24, 32))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 8), title_text, fill=(230, 235, 245))
    panel.paste(pil_img, (0, 30))
    return panel


def extract_and_visualize_features(model, checkpoint_path, left_img_path, right_img_path, output_dir, scale=2.0, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(">>> STEREOINR FEATURE MAP EXTRACTION & ARCHITECTURAL ANALYSIS")
    print(f" Target Device: {device}")
    print(f" Left Image:    {left_img_path}")
    print(f" Right Image:   {right_img_path}")
    print(f" Output Folder: {output_dir}")
    print("=" * 80)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights: {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["params"] if "params" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        print("Model weights loaded with strict=True!")
    else:
        print(f"[Warning] Checkpoint {checkpoint_path} not found. Running with initialized weights.")

    model = model.to(device)
    model.eval()

    # Register hooks
    hooks = FeatureExtractorHook()
    hook_handles = []

    # 1. Shallow convolution
    hook_handles.append(model.conv_first.register_forward_hook(hooks.hook_fn("1_conv_first")))

    # 2. Key RHAG Stages (12 stages total: 0, 3, 7, 11)
    if hasattr(model, "layers") and len(model.layers) >= 12:
        hook_handles.append(model.layers[0].register_forward_hook(hooks.hook_fn("2_rhag_stage_1")))
        hook_handles.append(model.layers[3].register_forward_hook(hooks.hook_fn("3_rhag_stage_4")))
        hook_handles.append(model.layers[7].register_forward_hook(hooks.hook_fn("4_rhag_stage_8")))
        hook_handles.append(model.layers[11].register_forward_hook(hooks.hook_fn("5_rhag_stage_12")))

    # 3. Post-body residual fusion
    hook_handles.append(model.conv_after_body.register_forward_hook(hooks.hook_fn("6_conv_after_body")))

    # 4. Pre-upsample features
    hook_handles.append(model.conv_before_upsample.register_forward_hook(hooks.hook_fn("7_conv_before_upsample")))

    # Prepare input images
    img_l = Image.open(left_img_path).convert("RGB")
    img_r = Image.open(right_img_path).convert("RGB")

    W_orig, H_orig = img_l.size
    crop_w, crop_h = min(256, W_orig), min(128, H_orig)
    img_l_crop = img_l.crop((0, 0, crop_w, crop_h))
    img_r_crop = img_r.crop((0, 0, crop_w, crop_h))

    # Low-resolution inputs
    lr_w, lr_h = max(16, int(crop_w / scale)), max(16, int(crop_h / scale))
    img_l_lr = img_l_crop.resize((lr_w, lr_h), Image.BICUBIC)
    img_r_lr = img_r_crop.resize((lr_w, lr_h), Image.BICUBIC)

    t_l = torch.tensor(np.array(img_l_lr)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t_r = torch.tensor(np.array(img_r_lr)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t_l = t_l.to(device)
    t_r = t_r.to(device)

    # Coordinate query grid
    h_out, w_out = crop_h, crop_w
    coord = make_coord((h_out, w_out)).unsqueeze(0).to(device)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2.0 / h_out
    cell[:, :, 1] *= 2.0 / w_out

    # Forward pass: StereoINR expects x of shape [B, 6, H, W], coord, cell
    x_input = torch.cat([t_l, t_r], dim=1)
    with torch.no_grad():
        sr_output = model(x_input, coord, cell)

    # Remove hooks
    for h in hook_handles:
        h.remove()

    print(f"Captured {len(hooks.features)} intermediate feature maps:")
    for k, v in sorted(hooks.features.items()):
        print(f"  - {k:<25}: shape = {list(v.shape)}")

    stages_to_plot = [
        ("1_conv_first", "Stage 0: Shallow Conv (3 -> 180 Channels)"),
        ("2_rhag_stage_1", "Stage 1: RHAG Block 1 (Local Edge & Texture)"),
        ("3_rhag_stage_4", "Stage 4: RHAG Block 4 (Mid-Level Patterns)"),
        ("4_rhag_stage_8", "Stage 8: RHAG Block 8 (Semantic Context)"),
        ("5_rhag_stage_12", "Stage 12: RHAG Block 12 (Stereo Cross-Aligned)"),
        ("6_conv_after_body", "Stage 13: Conv After Body (Deep Residual Fusion)"),
        ("7_conv_before_upsample", "Stage 14: Pre-Upsample Conditioning")
    ]

    panel_w, panel_h = 320, 180
    grid_w = panel_w * 3
    grid_h = panel_h * len(stages_to_plot) + 60

    master_canvas = Image.new("RGB", (grid_w, grid_h), color=(15, 18, 25))
    master_draw = ImageDraw.Draw(master_canvas)
    master_draw.text((20, 15), "StereoINR Layer-Wise Feature Map Progression (Input -> RHAG 1-12 -> DGASU)", fill=(255, 255, 255))
    master_draw.text((20, 35), "Col 1: Mean Activation (Viridis) | Col 2: Max Salient Edge (Inferno) | Col 3: PCA 3-Component RGB Projection", fill=(170, 185, 210))

    y_offset = 60
    for i, (k, label) in enumerate(stages_to_plot):
        if k not in hooks.features:
            continue
        feat = hooks.features[k]
        if feat.dim() == 4:
            feat_l = feat[0] # Left view
        elif feat.dim() == 3: # (N, L, C)
            feat_l = feat[0].transpose(0, 1).reshape(-1, lr_h, lr_w)
        else:
            feat_l = feat

        if feat_l.dim() == 3:
            # 1. Mean channel activation (normalized to [0, 1])
            mean_2d = feat_l.mean(dim=0)
            m_min, m_max = mean_2d.min(), mean_2d.max()
            mean_norm = (mean_2d - m_min) / (m_max - m_min + 1e-8)
            rgb_mean = colormap_viridis(mean_norm)

            # 2. Maximum activation channel (sharpest edge detector)
            max_2d = feat_l.abs().max(dim=0)[0]
            mx_min, mx_max = max_2d.min(), max_2d.max()
            max_norm = (max_2d - mx_min) / (mx_max - mx_min + 1e-8)
            rgb_max = colormap_inferno(max_norm)

            # 3. PCA 3-Component RGB projection
            rgb_pca = pca_projection(feat_l, n_components=3)

            # Build panels
            p1 = create_annotated_panel(rgb_mean, f"{label} [Mean]", panel_w, panel_h)
            p2 = create_annotated_panel(rgb_max, f"Max Salient Edge Channel", panel_w, panel_h)
            p3 = create_annotated_panel(rgb_pca, f"PCA RGB Disentanglement", panel_w, panel_h)

            master_canvas.paste(p1, (0, y_offset))
            master_canvas.paste(p2, (panel_w, y_offset))
            master_canvas.paste(p3, (panel_w * 2, y_offset))

            # Save individual stage preview as well
            single_stage = Image.new("RGB", (grid_w, panel_h), color=(15, 18, 25))
            single_stage.paste(p1, (0, 0))
            single_stage.paste(p2, (panel_w, 0))
            single_stage.paste(p3, (panel_w * 2, 0))
            single_stage.save(os.path.join(output_dir, f"{k}_feature_analysis.png"))

            y_offset += panel_h

    output_plot_path = os.path.join(output_dir, "StereoINR_Feature_Maps_All_Stages.png")
    master_canvas.save(output_plot_path, quality=95)
    print(f"\nSUCCESS: Master feature map grid saved to: {output_plot_path}")
    return output_plot_path


def build_model():
    kwargs = dict(
        upscale=4, in_chans=3, img_size=[32, 96], window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, drop_path_rate=0.1, img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], mlp_ratio=2,
        upsampler='continuous', resi_connection='1conv'
    )
    return StereoINR(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and Analyze StereoINR Feature Maps")
    parser.add_argument("--checkpoint", type=str, default="experiments/pretrained_models/StereoINR/StereoINR.pth")
    parser.add_argument("--left_img", type=str, default="demo/0001_L.png")
    parser.add_argument("--right_img", type=str, default="demo/0001_R.png")
    parser.add_argument("--output_dir", type=str, default="assets/feature_maps")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Fallback to test dataset if demo is not found
    if not os.path.exists(args.left_img):
        candidate_l = "datasets/StereoSR/test/Flickr1024/hr/0001/hr0.png"
        candidate_r = "datasets/StereoSR/test/Flickr1024/hr/0001/hr1.png"
        if os.path.exists(candidate_l):
            args.left_img = candidate_l
            args.right_img = candidate_r

    # Fallback for checkpoint
    if not os.path.exists(args.checkpoint):
        candidate_ckpt = r"d:\Gitrepo\Active Stereo RL\StereoINR.pth"
        if os.path.exists(candidate_ckpt):
            args.checkpoint = candidate_ckpt

    model = build_model()
    extract_and_visualize_features(
        model=model,
        checkpoint_path=args.checkpoint,
        left_img_path=args.left_img,
        right_img_path=args.right_img,
        output_dir=args.output_dir,
        scale=args.scale,
        device=args.device
    )
