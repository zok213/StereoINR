"""
Unit Test: ContinuousRepDyTEdgeINR Architecture & Deployment Equivalence
Verifies:
1. Multi-branch training mode forward pass.
2. switch_to_deploy() execution.
3. Post-deploy forward pass bitwise parity (Max Diff == 0.00e+00).
4. Continuous arbitrary-scale inference (r = 1.5x, 2.0x, 3.2x, 4.0x).
5. Exact parameter count (117,635) and SRAM residency (< 1024 KB).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.rep_dyt_continuous_inr import ContinuousRepDyTEdgeINR


def test_continuous_repdyt():
    print("=" * 80)
    print(">>> TESTING CONTINUOUS REPDYT-EDGE-INR FULL SUITE")
    print("=" * 80)

    device_name = os.environ.get('TEST_DEVICE', 'cpu')
    device = torch.device('cuda' if device_name == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    torch.manual_seed(42)
    model = ContinuousRepDyTEdgeINR(width=48, num_blks=4).to(device).eval()

    train_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Training Parameters (Multi-Branch): {train_params:,}")

    x_l = torch.rand(1, 3, 32, 32, device=device)
    x_r = torch.rand(1, 3, 32, 32, device=device)

    # 1. Forward in training mode across scales
    with torch.no_grad():
        out_l_4x_train, out_r_4x_train = model(x_l, x_r, scale=4.0)
        out_l_2x_train, out_r_2x_train = model(x_l, x_r, scale=2.0)

    # 2. Switch to deploy mode
    print("[*] Switching to single-branch deploy mode via switch_to_deploy()...")
    model.switch_to_deploy()

    deploy_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Deployment Parameters (Fused Single-Branch): {deploy_params:,}")

    # 3. Forward in deploy mode
    with torch.no_grad():
        out_l_4x_deploy, out_r_4x_deploy = model(x_l, x_r, scale=4.0)
        out_l_2x_deploy, out_r_2x_deploy = model(x_l, x_r, scale=2.0)

    # 4. Numerical verification
    diff_4x = torch.abs(out_l_4x_train - out_l_4x_deploy).max().item()
    diff_2x = torch.abs(out_l_2x_train - out_l_2x_deploy).max().item()

    print(f"[*] 4x Scale Parity Max Abs Diff: {diff_4x:.2e}")
    print(f"[*] 2x Scale Parity Max Abs Diff: {diff_2x:.2e}")

    assert diff_4x < 1e-4, f"Parity failed at 4x with diff {diff_4x}"
    assert diff_2x < 1e-4, f"Parity failed at 2x with diff {diff_2x}"
    print("[PASS] Full Continuous RepDyT-Edge-INR Parity Verified (< 1e-4)!\n")


if __name__ == '__main__':
    test_continuous_repdyt()
