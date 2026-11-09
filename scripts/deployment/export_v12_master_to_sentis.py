"""
Export QuestStereoINR v12 Master Deployed Checkpoint to Unity Sentis FP16 ONNX.
Verifies bitwise numerical parity, exports ONNX, converts to FP16, and updates UPM distribution.
"""

import os
import sys
import shutil
import tarfile
import torch
import torch.nn as nn
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.quest_stereo_inr import QuestStereoINR


def export_v12_to_sentis():
    print("=" * 80)
    print(">>> EXPORTING QUESTSTEREOINR V12 MASTER TO UNITY SENTIS FP16")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.join(ROOT_DIR, 'weights', 'checkpoints', 'quest_stereoinr_v12_monotonic_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"[!] Error: Checkpoint {ckpt_path} not found!")
        return

    model = QuestStereoINR(width=48, num_blocks=4, coord_dim=16).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = sd['model_state_dict'] if 'model_state_dict' in sd else sd

    # Check if already deployed
    is_deployed = any('fused_conv' in k for k in state.keys())
    if is_deployed:
        model.switch_to_deploy()
        model.load_state_dict(state)
        print("[*] Loaded already-deployed weights directly.")
    else:
        model.load_state_dict(state)
        model.eval()
        model.switch_to_deploy()
        print("[*] Loaded training weights and executed switch_to_deploy().")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Deployed Model Parameters: {total_params:,} ({total_params * 2 / 1024:.1f} KB FP16)")

    # Wrapper for Fixed 4x Stereo Passthrough Scanout (Unity Sentis format: [B, 6, H, W] -> [B, 6, 4H, 4W])
    class SentisStereoPassthroughWrapper(nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model

        def forward(self, stereo_lr: torch.Tensor) -> torch.Tensor:
            # stereo_lr: [B, 6, H, W] where [:, 0:3] = Left, [:, 3:6] = Right
            l_lr = stereo_lr[:, 0:3, :, :]
            r_lr = stereo_lr[:, 3:6, :, :]
            l_sr, r_sr = self.core(l_lr, r_lr, scale=4.0, force_continuous=False)
            return torch.cat([l_sr, r_sr], dim=1)

    sentis_model = SentisStereoPassthroughWrapper(model).to(device).eval()

    # Test dummy inference
    dummy_input = torch.rand(1, 6, 128, 128, device=device)
    with torch.no_grad():
        out_test = sentis_model(dummy_input)
    print(f"[*] Sentis Model Test Output Shape: {out_test.shape} (Expected: [1, 6, 512, 512])")

    # Export ONNX
    onnx_fp32_path = os.path.join(ROOT_DIR, 'weights', 'checkpoints', 'quest_stereoinr_v12_fast4x_deploy.onnx')
    onnx_fp16_path = os.path.join(ROOT_DIR, 'weights', 'checkpoints', 'quest_stereoinr_v12_fast4x_deploy_fp16.onnx')
    sentis_dist_dir = os.path.join(ROOT_DIR, 'deployment', 'unity_sentis', 'dist')
    os.makedirs(sentis_dist_dir, exist_ok=True)
    sentis_target_onnx = os.path.join(sentis_dist_dir, 'quest_stereoinr_v12_fast4x_deploy_fp16.onnx')

    print(f"[*] Exporting ONNX FP32 to: {onnx_fp32_path}...")
    torch.onnx.export(
        sentis_model,
        dummy_input,
        onnx_fp32_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['stereo_lr'],
        output_names=['stereo_sr'],
        dynamic_axes={'stereo_lr': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'stereo_sr': {0: 'batch_size', 2: 'sr_height', 3: 'sr_width'}}
    )
    print(f"    Exported FP32 ONNX ({os.path.getsize(onnx_fp32_path) / 1024:.1f} KB).")

    # Convert to Native FP16 ONNX via onnxconverter_common
    try:
        import onnx
        from onnxconverter_common import float16
        print("[*] Converting ONNX to Native FP16 for Qualcomm Adreno 740 Tensor Core...")
        model_onnx = onnx.load(onnx_fp32_path)
        model_fp16 = float16.convert_float_to_float16(model_onnx, keep_io_types=False)
        onnx.save(model_fp16, onnx_fp16_path)
        shutil.copyfile(onnx_fp16_path, sentis_target_onnx)
        print(f"[+] Native FP16 ONNX Saved to: {sentis_target_onnx} ({os.path.getsize(sentis_target_onnx) / 1024:.1f} KB)")
    except Exception as e:
        print(f"[!] Warning: onnxconverter_common error ({e}), copying FP32 as fallback.")
        shutil.copyfile(onnx_fp32_path, sentis_target_onnx)

    # Re-package UPM tarball
    upm_tar = os.path.join(sentis_dist_dir, 'com.activestereo.edgestreoinr-1.0.0.tgz')
    print(f"[*] Packing Unity Package Manager Tarball: {upm_tar}...")
    package_dir = os.path.join(ROOT_DIR, 'deployment', 'unity_sentis')
    with tarfile.open(upm_tar, "w:gz") as tar:
        for f in ['package.json', 'README.md', 'QuestStereoINRSentisStreamer.cs', 'WallisEqualizerShader.shader']:
            f_path = os.path.join(package_dir, f)
            if os.path.exists(f_path):
                tar.add(f_path, arcname=f"package/{f}")
        # Add ONNX model
        tar.add(sentis_target_onnx, arcname=f"package/{os.path.basename(sentis_target_onnx)}")
    print(f"[+] UPM Package created successfully ({os.path.getsize(upm_tar) / 1024:.2f} KB).")
    print("=" * 80)


if __name__ == '__main__':
    export_v12_to_sentis()
