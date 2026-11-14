"""
Lightweight CPU Checkpoint Auditor
==================================
Runs on CPU to avoid any GPU contention with background test suites.
Inspects weights/checkpoints/stereoinr_scale_liberated_improved.pth
"""

import os
import sys
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.stereoinr_scale_liberated import (
    ScaleLiberatedStereoINR,
    DynamicTanh
)


def audit_checkpoint():
    ckpt_path = os.path.join(ROOT_DIR, 'weights', 'checkpoints', 'stereoinr_scale_liberated_improved.pth')
    print(f"[*] Auditing checkpoint: {ckpt_path}")
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    file_size = os.path.getsize(ckpt_path)
    print(f"[*] Checkpoint size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"[*] Checkpoint keys: {list(ckpt.keys())}")

    # Metadata audit
    print(f"[*] Checkpoint Epoch: {ckpt.get('epoch')}")
    print(f"[*] Checkpoint PSNR: {ckpt.get('psnr')}")
    print(f"[*] Checkpoint MAE: {ckpt.get('mae')}")
    print(f"[*] Checkpoint Parity Diff: {ckpt.get('parity_diff')}")
    print(f"[*] Checkpoint Embed Dim: {ckpt.get('embed_dim')}")
    print(f"[*] Checkpoint Num Blocks: {ckpt.get('num_blocks')}")
    print(f"[*] Checkpoint Res Scale: {ckpt.get('res_scale')}")

    # Model instantiation on CPU
    model = ScaleLiberatedStereoINR(
        embed_dim=ckpt.get('embed_dim', 144),
        num_blocks=ckpt.get('num_blocks', 8),
        res_scale=ckpt.get('res_scale', 0.2)
    ).cpu()

    # Load training weights
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    train_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model train parameters: {train_params:,}")
    assert train_params == 4799956, f"Expected 4,799,956 train params, got {train_params}"

    # Audit Dynamic Tanh alphas in checkpoint
    dyt_layers = []
    max_alpha = 0.0
    for name, m in model.named_modules():
        if isinstance(m, DynamicTanh):
            alpha_abs = m.alpha.detach().abs().max().item()
            gamma = m.lipschitz_constant
            dyt_layers.append((name, alpha_abs, gamma))
            if gamma > max_alpha:
                max_alpha = gamma

    print(f"[*] Dynamic Tanh layers found: {len(dyt_layers)}")
    print(f"[*] Maximum empirical Lipschitz constant Gamma across all DyT layers: {max_alpha:.4f}")
    assert max_alpha <= 2.0, f"Lipschitz Gamma violated! Max={max_alpha}"

    # Convert to deploy mode
    model.switch_to_deploy()
    deploy_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model deploy parameters: {deploy_params:,}")
    assert deploy_params == 4468180, f"Expected 4,468,180 deploy params, got {deploy_params}"

    # Compare converted model state_dict with saved deploy_state_dict
    saved_deploy_sd = ckpt['deploy_state_dict']
    current_deploy_sd = model.state_dict()

    missing_in_saved = set(current_deploy_sd.keys()) - set(saved_deploy_sd.keys())
    missing_in_current = set(saved_deploy_sd.keys()) - set(current_deploy_sd.keys())

    assert len(missing_in_saved) == 0, f"Keys missing in saved deploy_state_dict: {missing_in_saved}"
    assert len(missing_in_current) == 0, f"Extra keys in saved deploy_state_dict: {missing_in_current}"

    # Tensor value comparison between converted weights and saved deploy_state_dict
    max_weight_diff = 0.0
    for k in current_deploy_sd.keys():
        diff = (current_deploy_sd[k].float() - saved_deploy_sd[k].float()).abs().max().item()
        if diff > max_weight_diff:
            max_weight_diff = diff

    print(f"[*] Maximum weight tensor difference between switch_to_deploy() and saved deploy_state_dict: {max_weight_diff:.2e}")
    assert max_weight_diff < 1e-6, f"Weight discrepancy between fused and saved deploy dict: {max_weight_diff}"

    print("[SUCCESS] Checkpoint CPU audit passed with 100% integrity!")


if __name__ == '__main__':
    audit_checkpoint()
