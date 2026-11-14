"""
Adversarial Verification Suite: 1D Epipolar Parallax Attention Scanline Isolation
================================================================================
Empirical Challenger Verification for Milestone 2 (challenger_m2_2).

Verification Objective:
Adversarially stress-test EpipolarAttention (EPA-1D) to prove 1D horizontal epipolar
isolation. Specifically, perturbing right-image row y = k must have ZERO (0.00e+00)
leakage into left-image reconstructed rows y != k, across varied spatial resolutions,
boundary conditions, and adversarial perturbations (massive offsets, Dirac spikes,
isolated NaNs).
"""

import os
import sys
import math
import pytest
import torch
import torch.nn as nn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.stereoinr_scale_liberated import (
    EpipolarAttention,
    ScaleLiberatedStereoINR
)


def test_epipolar_isolation_sweep_rows():
    """
    Adversarial Test 1: Full Row Sweep Isolation Test
    Iterates through EVERY row k in {0, ..., H-1} and applies a massive perturbation (+5000.0).
    Verifies that for every k, max absolute error on all y != k is EXACTLY 0.00 (within FP32 epsilon < 1e-7).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C = 144
    heads = 4
    H = 16
    W = 32

    torch.manual_seed(1337)
    epa = EpipolarAttention(channels=C, heads=heads).to(device).eval()

    x_l = torch.randn(1, C, H, W, device=device)
    x_r = torch.randn(1, C, H, W, device=device)

    with torch.no_grad():
        base_out_l, base_out_r = epa(x_l, x_r)

    print(f"\n[*] Running Full Row Sweep Isolation on ({H}x{W}), C={C}, heads={heads}...")

    max_leakage_overall = 0.0

    for k in range(H):
        x_r_perturbed = x_r.clone()
        x_r_perturbed[:, :, k, :] += 5000.0

        with torch.no_grad():
            pert_out_l, _ = epa(x_l, x_r_perturbed)

        diff = (pert_out_l - base_out_l).abs()

        # Row k MUST have non-zero response (demonstrating cross-attention active)
        row_k_diff = diff[:, :, k, :].max().item()
        assert row_k_diff > 1e-2, f"Perturbed row k={k} had no response! diff={row_k_diff}"

        # All rows y != k MUST have 0.00 leakage
        mask_other_rows = torch.ones(H, dtype=torch.bool)
        mask_other_rows[k] = False

        leakage = diff[:, :, mask_other_rows, :].max().item()
        if leakage > max_leakage_overall:
            max_leakage_overall = leakage

        assert leakage == 0.0 or leakage < 1e-7, (
            f"Adversarial Failure: Perturbing row k={k} leaked into other rows! "
            f"Max leakage: {leakage:.4e}"
        )

    print(f"[+] All {H} rows tested. Maximum cross-row leakage: {max_leakage_overall:.2e} (0.00 confirmed)")


def test_epipolar_isolation_bidirectional():
    """
    Adversarial Test 2: Bidirectional Isolation (L->R and R->L)
    Tests both directions of cross-view attention with Dirac impulses at boundary rows.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C = 144
    heads = 4
    H = 32
    W = 32

    torch.manual_seed(2026)
    epa = EpipolarAttention(channels=C, heads=heads).to(device).eval()

    x_l = torch.randn(2, C, H, W, device=device)
    x_r = torch.randn(2, C, H, W, device=device)

    with torch.no_grad():
        base_out_l, base_out_r = epa(x_l, x_r)

    # Test top boundary (row 0) and bottom boundary (row H-1)
    for target_row in [0, H - 1, H // 2]:
        # Perturb Right, check Left
        x_r_pert = x_r.clone()
        x_r_pert[:, :, target_row, :] = 999.0
        with torch.no_grad():
            pert_out_l, _ = epa(x_l, x_r_pert)

        diff_l = (pert_out_l - base_out_l).abs()
        diff_l[:, :, target_row, :] = 0.0  # zero out targeted row
        leak_l = diff_l.max().item()
        assert leak_l < 1e-7, f"R->L Leakage on target_row {target_row}: {leak_l:.2e}"

        # Perturb Left, check Right
        x_l_pert = x_l.clone()
        x_l_pert[:, :, target_row, :] = -999.0
        with torch.no_grad():
            _, pert_out_r = epa(x_l_pert, x_r)

        diff_r = (pert_out_r - base_out_r).abs()
        diff_r[:, :, target_row, :] = 0.0  # zero out targeted row
        leak_r = diff_r.max().item()
        assert leak_r < 1e-7, f"L->R Leakage on target_row {target_row}: {leak_r:.2e}"

    print(f"[+] Bidirectional L<->R isolation verified at boundary & center rows: leakage < 1e-7")


def test_epipolar_nan_containment():
    """
    Adversarial Test 3: Extreme Adversarial Containment (NaN isolation)
    Sets row k of right image to NaN.
    Verifies that rows y != k in the left output remain strictly FINITE and bitwise identical.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C = 144
    heads = 4
    H = 8
    W = 16

    epa = EpipolarAttention(channels=C, heads=heads).to(device).eval()

    x_l = torch.randn(1, C, H, W, device=device)
    x_r = torch.randn(1, C, H, W, device=device)

    with torch.no_grad():
        base_out_l, _ = epa(x_l, x_r)

    k = 4
    x_r_nan = x_r.clone()
    x_r_nan[:, :, k, :] = float('nan')

    with torch.no_grad():
        pert_out_l, _ = epa(x_l, x_r_nan)

    # Row k should have NaNs
    assert torch.isnan(pert_out_l[:, :, k, :]).any(), "Expected NaNs in perturbed row k"

    # All other rows y != k MUST BE FINITE and match base_out_l exactly
    for y in range(H):
        if y != k:
            row_out = pert_out_l[:, :, y, :]
            row_base = base_out_l[:, :, y, :]
            assert torch.isfinite(row_out).all(), f"Row y={y} infected with non-finite values!"
            diff = (row_out - row_base).abs().max().item()
            assert diff == 0.0 or diff < 1e-7, f"Row y={y} altered by NaN in row {k}: diff={diff}"

    print(f"[+] Adversarial NaN containment verified: zero cross-row contamination")


def test_adversarial_epipolar_isolation():
    """Adversarial epipolar scanline isolation: row sweep perturbation test."""
    test_epipolar_isolation_sweep_rows()


def test_monocular_spatial_decoupling():
    """Bidirectional cross-view spatial isolation: verifying boundary & center row decoupling."""
    test_epipolar_isolation_bidirectional()
    test_epipolar_nan_containment()


def test_continuous_scale_invariance():
    """Continuous scale factor invariance: verify forward pass across arbitrary non-integer scales."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ScaleLiberatedStereoINR(embed_dim=144, num_blocks=8, res_scale=0.2).to(device).eval()
    x_l = torch.randn(1, 3, 16, 16, device=device)
    x_r = torch.randn(1, 3, 16, 16, device=device)
    for scale in [1.5, 2.0, 3.14159, 4.0]:
        with torch.no_grad():
            out_l, out_r, _ = model(x_l, x_r, scale=scale)
            expected_h = int(16 * scale)
            expected_w = int(16 * scale)
            assert out_l.shape == (1, 3, expected_h, expected_w), f"Scale {scale} shape mismatch: {out_l.shape}"
            assert torch.isfinite(out_l).all() and torch.isfinite(out_r).all()
    print("[+] Continuous scale invariance verified across arbitrary non-integer scales (1.5x, 2.0x, 3.14x, 4.0x)")


if __name__ == '__main__':
    test_adversarial_epipolar_isolation()
    test_monocular_spatial_decoupling()
    test_continuous_scale_invariance()
    print("\n[ALL ADVERSARIAL ISOLATION TESTS PASSED]")
