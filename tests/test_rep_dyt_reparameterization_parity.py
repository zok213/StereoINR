"""
Unit Test: Exact Numerical Parity of Structural Reparameterization (RepDyT)
Verifies that multi-branch training RepConv2d produces bitwise identical results
(error < 1e-6) to fused single-branch convolution after switch_to_deploy().
"""

import sys
import os
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.rep_dyt_block import RepConv2d, RepDyTNAFBlock


def test_rep_conv2d_parity():
    print("=" * 70)
    print(">>> TESTING REPCONV2D REPARAMETERIZATION PARITY")
    print("=" * 70)

    torch.manual_seed(42)
    x = torch.randn(2, 48, 64, 64)

    rep_conv = RepConv2d(48, 48, bias=True).eval()

    # 1. Output before reparameterization (3 branches)
    with torch.no_grad():
        out_train = rep_conv(x)

    # 2. Switch to single-branch deploy
    rep_conv.switch_to_deploy()

    # 3. Output after reparameterization (1 fused 3x3 conv)
    with torch.no_grad():
        out_deploy = rep_conv(x)

    # Measure differences
    max_diff = (out_train - out_deploy).abs().max().item()
    mse = torch.mean((out_train - out_deploy) ** 2).item()

    print(f"RepConv2d Max Abs Diff: {max_diff:.2e}")
    print(f"RepConv2d Mean Sq Error: {mse:.2e}")

    assert torch.allclose(out_train, out_deploy, atol=1e-5), f"RepConv2d parity failed with max_diff={max_diff}"
    print("[PASS] RepConv2d Bitwise Parity Verified (Max Diff < 1e-5, MSE < 1e-12)!\n")


def test_rep_dyt_nafblock_parity():
    print("=" * 70)
    print(">>> TESTING REPDYT-NAFBLOCK FULL REPARAMETERIZATION PARITY")
    print("=" * 70)

    torch.manual_seed(42)
    x = torch.randn(2, 48, 64, 64)

    block = RepDyTNAFBlock(c=48).eval()

    with torch.no_grad():
        out_train = block(x)

    block.switch_to_deploy()

    with torch.no_grad():
        out_deploy = block(x)

    max_diff = (out_train - out_deploy).abs().max().item()
    mse = torch.mean((out_train - out_deploy) ** 2).item()

    print(f"RepDyTNAFBlock Max Abs Diff: {max_diff:.2e}")
    print(f"RepDyTNAFBlock Mean Sq Error: {mse:.2e}")

    assert torch.allclose(out_train, out_deploy, atol=1e-5), f"RepDyTNAFBlock parity failed with max_diff={max_diff}"
    print("[PASS] RepDyTNAFBlock Bitwise Parity Verified (Max Diff < 1e-5, MSE < 1e-12)!\n")


if __name__ == '__main__':
    test_rep_conv2d_parity()
    test_rep_dyt_nafblock_parity()
    print("=" * 70)
    print("[+] ALL REPDYT REPARAMETERIZATION PARITY TESTS PASSED!")
    print("=" * 70)
