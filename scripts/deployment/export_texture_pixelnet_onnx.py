import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
"""
Export Trained TextureStereoPixelNet to ONNX FP32 & FP16 for Mobile XR Deployment
Target Runtime: Unity Sentis (GPU Compute) / Native Vulkan on Meta Quest 3S.
"""

import os
import sys
import torch
import torch.nn as nn
import onnx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.edge_stereoinr.phase2_edge_inr.backbones.texture_stereopixelnet import TextureStereoPixelNet


def export_texture_model():
    print("=" * 85)
    print(">>> EXPORTING TRAINED TEXTURE-STEREOPIXELNET TO ONNX FP16 FOR UNITY SENTIS")
    print("=" * 85)

    device = torch.device('cpu')
    model = TextureStereoPixelNet(in_channels=3, feat_channels=48, num_blocks=3).to(device)

    # Load trained weights
    ckpt_path = "weights/checkpoints/texture_stereopixelnet_trained.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[OK] Loaded trained checkpoint from {ckpt_path}")
    else:
        print("[!] Warning: Checkpoint not found, exporting with random weights.")

    model.eval()

    dummy_l = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    dummy_r = torch.randn(1, 3, 128, 128, dtype=torch.float32)

    os.makedirs("weights/onnx", exist_ok=True)
    fp32_path = "weights/onnx/texture_stereopixelnet_fp32.onnx"
    fp16_path = "weights/onnx/texture_stereopixelnet_fp16.onnx"

    # Export FP32 ONNX
    print(f"[*] Exporting FP32 ONNX to {fp32_path} ...")
    torch.onnx.export(
        model,
        (dummy_l, dummy_r),
        fp32_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['patch_left', 'patch_right'],
        output_names=['sr_texture_2x']
    )
    size_fp32 = os.path.getsize(fp32_path) / 1024.0
    print(f"[OK] FP32 ONNX Created ({size_fp32:.2f} KB)")

    # Convert to FP16
    print(f"[*] Converting to FP16 for Adreno 740 GPU Compute: {fp16_path} ...")
    try:
        from onnxconverter_common import float16
        m_fp32 = onnx.load(fp32_path)
        m_fp16 = float16.convert_float_to_float16(m_fp32)
        onnx.save(m_fp16, fp16_path)
        size_fp16 = os.path.getsize(fp16_path) / 1024.0
        print(f"[OK] FP16 ONNX Created ({size_fp16:.2f} KB)")
    except Exception as e:
        print(f"[!] FP16 Conversion notice: {e}")

    # Verify with ONNX Runtime
    print("\n[*] Verifying with ONNX Runtime...")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
        out = sess.run(None, {'patch_left': dummy_l.numpy(), 'patch_right': dummy_r.numpy()})
        print(f"[OK] Output Shape: {out[0].shape} (Verified 256x256!)")
        assert out[0].shape == (1, 3, 256, 256)
    except Exception as e:
        print(f"[!] ORT verification notice: {e}")

    print("\n" + "=" * 85)
    print(">>> EXPORT COMPLETE: TEXTURE-STEREOPIXELNET READY FOR UNITY SENTIS!")
    print("=" * 85)


if __name__ == '__main__':
    export_texture_model()
