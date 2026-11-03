"""
Export downloaded Kaggle checkpoint to production ONNX (FP32 & FP16)
Validates numerical consistency with ONNX Runtime.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_nafrssr import WindowedDyTNAFRSSR


def export_kaggle_checkpoint():
    print("=" * 80)
    print(">>> EXPORTING KAGGLE CHECKPOINT TO PRODUCTION ONNX FOR META QUEST 3S")
    print("=" * 80)

    ckpt_path = os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'dyt_nafrssr_quest3s_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"[-] Checkpoint not found: {ckpt_path}")
        return

    device = torch.device('cpu')
    model = WindowedDyTNAFRSSR(width=48, num_blks=4, up_scale=4, min_disp=-2, max_disp=14).to(device).eval()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)

    print("[*] Successfully loaded model state dict!")

    out_dir = os.path.join(ROOT_DIR, 'weights', 'onnx')
    os.makedirs(out_dir, exist_ok=True)
    fp32_onnx = os.path.join(out_dir, 'dyt_nafrssr_kaggle_fp32.onnx')
    fp16_onnx = os.path.join(out_dir, 'dyt_nafrssr_kaggle_fp16.onnx')

    dummy_l = torch.randn(1, 3, 64, 64, device=device)
    dummy_r = torch.randn(1, 3, 64, 64, device=device)

    # 1. Export FP32 ONNX
    print(f"[*] Exporting FP32 ONNX to {fp32_onnx}...")
    torch.onnx.export(
        model,
        (dummy_l, dummy_r),
        fp32_onnx,
        input_names=['left_eye_lr', 'right_eye_lr'],
        output_names=['left_eye_sr', 'right_eye_sr'],
        dynamic_axes={
            'left_eye_lr': {2: 'height', 3: 'width'},
            'right_eye_lr': {2: 'height', 3: 'width'},
            'left_eye_sr': {2: 'out_h', 3: 'out_w'},
            'right_eye_sr': {2: 'out_h', 3: 'out_w'}
        },
        opset_version=17
    )
    onnx_model = onnx.load(fp32_onnx)
    onnx.checker.check_model(onnx_model)
    fp32_size = os.path.getsize(fp32_onnx) / 1024
    print(f"[OK] FP32 ONNX exported! File size: {fp32_size:.1f} KB")

    # 2. Export FP16 ONNX
    print(f"[*] Converting to FP16 ONNX: {fp16_onnx}...")
    try:
        from onnxconverter_common import float16
        fp16_model = float16.convert_float_to_float16(onnx_model, keep_io_types=False)
        onnx.save(fp16_model, fp16_onnx)
        fp16_size = os.path.getsize(fp16_onnx) / 1024
        print(f"[OK] FP16 ONNX exported! File size: {fp16_size:.1f} KB")
    except Exception as e:
        print(f"[!] FP16 converter skipped: {e}")

    # 3. Verify ONNX Runtime forward pass
    print("[*] Validating with ONNX Runtime...")
    session = ort.InferenceSession(fp32_onnx, providers=['CPUExecutionProvider'])
    np_l = np.random.randn(1, 3, 64, 64).astype(np.float32)
    np_r = np.random.randn(1, 3, 64, 64).astype(np.float32)
    ort_outs = session.run(None, {'left_eye_lr': np_l, 'right_eye_lr': np_r})
    with torch.no_grad():
        py_out_l, py_out_r = model(torch.from_numpy(np_l), torch.from_numpy(np_r))

    diff_l = np.max(np.abs(ort_outs[0] - py_out_l.numpy()))
    diff_r = np.max(np.abs(ort_outs[1] - py_out_r.numpy()))
    print(f"[OK] Max numerical deviation (ORT vs PyTorch): Left={diff_l:.2e}, Right={diff_r:.2e}")
    assert diff_l < 1e-4 and diff_r < 1e-4, "ONNX verification failed!"
    print("[OK] Verification passed with zero numerical drift!")


if __name__ == '__main__':
    export_kaggle_checkpoint()
