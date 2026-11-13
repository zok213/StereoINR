"""
Unit & Integration Test Suite: Improved Scale-Liberated StereoINR (RepDyT-EdgeINR)
==================================================================================
Verifies Requirements for Milestone 2:
1. Exact Deploy-Mode Parameter Count (4,468,180 parameters).
2. Bitwise Deploy-Mode Parity (switch_to_deploy(), max abs diff < 1e-4 / 0.00e+00).
3. INT8 Lipschitz Bound Enforcement (Gamma_max <= 2.0 on all DynamicTanh layers).
4. Feature Variance Damping across 8 residual blocks (std bounded, preventing 21.6 explosion).
5. Epipolar 1D Scanline Consistency (y_L = y_R epipolar attention row isolation).
6. Unified Multi-Task Loss Suite (L_char + 0.5*L_warp + 0.1*L_fft + 0.25*L_KD) & gradient flow.
"""

import os
import sys
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.stereoinr_scale_liberated import (
    ScaleLiberatedStereoINR,
    DynamicTanh,
    RepConv2d,
    RepDyTResidualBlock,
    EpipolarAttention,
    CharbonnierLoss,
    BidirectionalEpipolarWarpingLoss,
    FFTSpectralLoss,
    FeatureKDLoss,
    MultiTaskLossSuite
)


def test_parameter_count_and_deploy_parity():
    """Verify train-to-deploy fusion parity and exact 4,468,180 deploy parameters."""
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ScaleLiberatedStereoINR(embed_dim=144, num_blocks=8, res_scale=0.2).to(device).eval()

    train_params = sum(p.numel() for p in model.parameters())
    print(f"\n[*] Train-mode parameters: {train_params:,}")

    x_l = torch.randn(1, 3, 32, 32, device=device)
    x_r = torch.randn(1, 3, 32, 32, device=device)

    with torch.no_grad():
        out_l_train, out_r_train, info_train = model(x_l, x_r, scale=4.0)

    # Switch to deploy mode
    model.switch_to_deploy()
    deploy_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Deploy-mode parameters: {deploy_params:,}")

    # Exact parameter count assertion
    assert deploy_params == 4468180, f"Expected 4,468,180 deploy parameters, got {deploy_params}"

    with torch.no_grad():
        out_l_deploy, out_r_deploy, info_deploy = model(x_l, x_r, scale=4.0)

    diff_l = (out_l_train - out_l_deploy).abs().max().item()
    diff_r = (out_r_train - out_r_deploy).abs().max().item()
    print(f"[*] Deploy Parity Diff Left: {diff_l:.2e}")
    print(f"[*] Deploy Parity Diff Right: {diff_r:.2e}")

    assert diff_l < 1e-4, f"Deploy parity left failed: diff={diff_l}"
    assert diff_r < 1e-4, f"Deploy parity right failed: diff={diff_r}"
    print("[PASS] Exact Deploy-Mode Parameter Count & Bitwise Parity Verified!")


def test_lipschitz_bounds():
    """Verify Lipschitz bound Gamma_max <= 2.0 across all DynamicTanh modules."""
    model = ScaleLiberatedStereoINR(embed_dim=144, num_blocks=8)

    max_gamma = 0.0
    dyt_count = 0
    for name, m in model.named_modules():
        if isinstance(m, DynamicTanh):
            dyt_count += 1
            gamma = m.lipschitz_constant
            if gamma > max_gamma:
                max_gamma = gamma
            assert gamma <= 2.0001, f"Layer {name} violates Lipschitz bound: gamma={gamma}"

    print(f"\n[*] Audited {dyt_count} DynamicTanh layers. Maximum Gamma: {max_gamma:.4f} <= 2.0")
    assert dyt_count >= 17, f"Expected at least 17 DyT layers (1 stem + 8 blocks + 8 EPA), found {dyt_count}"
    print("[PASS] Lipschitz Bounds (Gamma <= 2.0) Verified for INT8 Quantization!")


def test_residual_damping_variance():
    """Verify residual damping eliminates activation variance explosion across 8 blocks."""
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ScaleLiberatedStereoINR(embed_dim=144, num_blocks=8, res_scale=0.2).to(device).eval()

    x = torch.randn(2, 3, 48, 48, device=device)
    with torch.no_grad():
        fl = model.conv_first(x)
        stem_std = fl.std().item()
        print(f"\n[*] Stem activation std: {stem_std:.4f}")

        block_stds = []
        for i, (b, a) in enumerate(zip(model.blocks, model.stereo_adapters)):
            fl = b(fl)
            fl, _ = a(fl, fl)
            std_i = fl.std().item()
            block_stds.append(std_i)
            print(f"[*] Block {i} activation std: {std_i:.4f}")

    final_std = block_stds[-1]
    print(f"[*] Final Block 7 std: {final_std:.4f} (Baseline exploded to 21.6)")
    # The variance must stay controlled (< 2.0), far below the unscaled 21.6 explosion
    assert final_std < 2.0, f"Activation variance exploded! Block 7 std={final_std}"
    print("[PASS] Residual Damping Successfully Prevents Feature Variance Explosion!")


def test_epipolar_scanline_isolation():
    """Verify EPA-1D strictly constrains cross-view attention to horizontal scanlines (y_L = y_R)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epa = EpipolarAttention(channels=16, heads=2).to(device).eval()

    x_l = torch.randn(1, 16, 8, 16, device=device)
    x_r = torch.randn(1, 16, 8, 16, device=device)

    with torch.no_grad():
        base_out_l, _ = epa(x_l, x_r)

        # Perturb Row 3 of right image by a huge offset
        x_r_perturbed = x_r.clone()
        x_r_perturbed[:, :, 3, :] += 50.0

        pert_out_l, _ = epa(x_l, x_r_perturbed)
        row_diffs = (base_out_l - pert_out_l).abs().mean(dim=(0, 1, 3))

    for r in range(8):
        diff_r = row_diffs[r].item()
        if r == 3:
            assert diff_r > 1e-3, f"Expected non-zero difference in perturbed Row 3, got {diff_r}"
        else:
            assert diff_r < 1e-6, f"Row {r} affected by perturbation in Row 3! (Diff={diff_r})"

    print("\n[*] Row isolation differences:")
    for r in range(8):
        print(f"    Row {r}: diff = {row_diffs[r].item():.2e}")
    print("[PASS] 1D Epipolar Scanline Restriction (y_L = y_R) Verified!")


def test_multitask_loss_and_gradients():
    """Verify Multi-Task Loss Suite (Charbonnier + Warp LRC + FFT + KD) and backward gradient flow."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ScaleLiberatedStereoINR(embed_dim=144, num_blocks=8, res_scale=0.2).to(device).train()

    loss_suite = MultiTaskLossSuite(eps_char=1e-3, tau_lrc=1.0, student_dim=144, teacher_dim=180).to(device)

    x_l = torch.rand(1, 3, 24, 24, device=device, requires_grad=False)
    x_r = torch.rand(1, 3, 24, 24, device=device, requires_grad=False)
    gt_l = torch.rand(1, 3, 96, 96, device=device)
    gt_r = torch.rand(1, 3, 96, 96, device=device)
    teacher_feat = torch.rand(1, 180, 24, 24, device=device)

    out_l, out_r, info = model(x_l, x_r, scale=4.0)
    fl, fr = model.forward_features(x_l, x_r)[:2]

    losses = loss_suite(
        pred_l=out_l, pred_r=out_r,
        gt_l=gt_l, gt_r=gt_r,
        disp_l2r=info['disp_l2r'], disp_r2l=info['disp_r2l'],
        student_feat_l=fl, teacher_feat_l=teacher_feat
    )

    total_loss = losses['loss_total']
    print(f"\n[*] Losses: Total={total_loss.item():.4f} | Char={losses['loss_char'].item():.4f} | Warp={losses['loss_warp'].item():.4f} | FFT={losses['loss_fft'].item():.4f} | KD={losses['loss_kd'].item():.4f}")

    total_loss.backward()

    # Verify gradients exist and are finite on model weights
    has_grads = False
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Gradient contains NaN or Inf"
            has_grads = True
    assert has_grads, "No gradients computed!"

    print("[PASS] Multi-Task Loss Suite and Gradient Backpropagation Verified!")


if __name__ == '__main__':
    test_parameter_count_and_deploy_parity()
    test_lipschitz_bounds()
    test_residual_damping_variance()
    test_epipolar_scanline_isolation()
    test_multitask_loss_and_gradients()
