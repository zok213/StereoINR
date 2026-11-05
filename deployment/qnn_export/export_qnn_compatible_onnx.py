"""
Qualcomm Snapdragon XR2 Gen 2 / Hexagon NPU ONNX Exporter for DyT-Edge-INR:
Exports the Pareto Model Soup into a static-shaped, QNN-compliant ONNX graph
optimized for Qualcomm Hexagon Direct HVX Vector Execution.

Target Specifications:
- Target Hardware: Qualcomm Hexagon NPU / Snapdragon XR2 Gen 2 (Meta Quest 3S)
- Quantization Target: Symmetric INT8 (Per-Channel Weights, Per-Tensor Activations)
- Static Tile Dimensions: (1, 3, 64, 64) per eye
- ONNX Opset: 17 (fully supported by Qualcomm QNN SDK v2.18+)
"""

import os, sys
import torch
import torch.nn as nn
import onnx

ROOT_DIR = r"d:\Gitrepo\Active Stereo RL"
sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_continuous_scale_inr import ContinuousDyTEdgeINR


class QNNFixedScaleWrapper(nn.Module):
    """
    Static wrapper for QNN NPU compilation:
    Fixes scale at 4.0x or pre-compiles the graph to avoid dynamic control flow.
    """
    def __init__(self, model: ContinuousDyTEdgeINR, fixed_scale: float = 4.0):
        super().__init__()
        self.model = model
        self.fixed_scale = float(fixed_scale)

    def forward(self, img_l: torch.Tensor, img_r: torch.Tensor):
        return self.model(img_l, img_r, scale=self.fixed_scale)


def export_qnn_onnx():
    device = torch.device('cpu')
    ckpt_path = os.path.join(ROOT_DIR, 'weights', 'kaggle_dual_t4', 'dyt_continuous_inr_quest3s_soup_best.pth')
    
    print("[*] Loading Model Soup checkpoint...")
    base_model = ContinuousDyTEdgeINR(width=48, num_blks=4).to(device).eval()
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    base_model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd, strict=True)
    
    wrapper = QNNFixedScaleWrapper(base_model, fixed_scale=4.0)
    
    dummy_l = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    dummy_r = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    
    out_dir = os.path.join(ROOT_DIR, 'weights', 'qnn_npu')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'dyt_edge_inr_qnn_tile64_4x.onnx')
    
    print(f"[*] Exporting static QNN-compliant ONNX to: {out_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_l, dummy_r),
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_l', 'input_r'],
        output_names=['output_l', 'output_r'],
    )
    
    # Validate with ONNX checker
    model_onnx = onnx.load(out_path)
    onnx.checker.check_model(model_onnx)
    print(f"[PASS] ONNX graph verified valid! File size: {os.path.getsize(out_path)/1024:.1f} KB")

if __name__ == '__main__':
    export_qnn_onnx()
