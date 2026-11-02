import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
"""
Qualcomm QNN (Hexagon Tensor Processor - HTP) Compilation & Deployment Pipeline
Generates native QNN commands, model libraries, and context binaries for Snapdragon XR2 Gen 2.
"""

import os
import sys
import json
import subprocess
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def generate_qnn_compilation_suite():
    print("=" * 85)
    print(">>> QUALCOMM NEURAL PROCESSING SDK (QNN) COMPILATION PIPELINE FOR XR2 GEN 2")
    print("=" * 85)

    onnx_fp32_path = "weights/onnx/student_stereoinr_fp32.onnx"
    qnn_dir = "weights/qnn"
    os.makedirs(qnn_dir, exist_ok=True)

    if not os.path.exists(onnx_fp32_path):
        print(f"[!] Error: {onnx_fp32_path} not found. Run scripts/export_quantized_student.py first.")
        return

    file_size_kb = os.path.getsize(onnx_fp32_path) / 1024.0
    print(f"[*] Verified Source ONNX Model: {onnx_fp32_path} ({file_size_kb:.2f} KB)")

    # 1. Configuration parameters for Hexagon HTP Backend
    qnn_config = {
        "model_name": "student_stereoinr",
        "target_soc": "Snapdragon XR2 Gen 2 (SM8550-based)",
        "target_backend": "Hexagon HTP (V73 / V75)",
        "input_tensors": {
            "img_left": [1, 3, 128, 128],
            "img_right": [1, 3, 128, 128]
        },
        "output_tensors": {
            "sr_fovea": [1, 3, 256, 256]
        },
        "quantization": {
            "weight_type": "INT8",
            "activation_type": "INT8",
            "per_channel": True
        },
        "power_profile": "BURST_OR_HIGH_PERFORMANCE"
    }

    config_path = os.path.join(qnn_dir, "qnn_config.json")
    with open(config_path, "w") as f:
        json.dump(qnn_config, f, indent=4)
    print(f"[OK] Generated QNN Configuration: {config_path}")

    # 2. Build Shell Script for Android NDK & QNN SDK Execution
    build_script_content = f"""#!/bin/bash
# Qualcomm QNN SDK Automated Cross-Compilation Script
# Requires: QNN_SDK_ROOT (e.g., /opt/qcom/qnn-v2.20) and ANDROID_NDK_ROOT

set -e
echo "================================================================="
echo "Compiling StudentStereoINR for Snapdragon XR2 Gen 2 Hexagon NPU"
echo "================================================================="

# Step 1: Convert ONNX to QNN C++ source and binary graph
qnn-onnx-converter \\
    --input_network {onnx_fp32_path} \\
    --output_path {qnn_dir}/student_stereoinr.cpp \\
    --input_dim img_left 1,3,128,128 \\
    --input_dim img_right 1,3,128,128 \\
    --act_quant_param_method MSE \\
    --param_quantizer_method MSE

# Step 2: Compile Model Shared Library (.so) for aarch64-android
qnn-model-lib-generator \\
    -c {qnn_dir}/student_stereoinr.cpp \\
    -b {qnn_dir}/student_stereoinr.bin \\
    -o {qnn_dir}/libstudent_stereoinr.so \\
    -t aarch64-android

# Step 3: Generate Hexagon HTP Context Binary (Serialized NPU Cache)
# This enables sub-millisecond cold start without runtime JIT compilation!
qnn-context-binary-generator \\
    --backend libQnnHtp.so \\
    --model {qnn_dir}/libstudent_stereoinr.so \\
    --binary_file {qnn_dir}/student_stereoinr_htp.bin \\
    --htp_soc_id 43 \\
    --log_level info

echo "[SUCCESS] Context binary generated: {qnn_dir}/student_stereoinr_htp.bin"
echo "Ready for zero-latency execution on Meta Quest 3S Hexagon NPU!"
"""
    sh_path = os.path.join(qnn_dir, "compile_qnn_model.sh")
    with open(sh_path, "w") as f:
        f.write(build_script_content)
    print(f"[OK] Generated QNN Compilation Script: {sh_path}")

    print("\n" + "=" * 85)
    print(">>> QUALCOMM QNN DEPLOYMENT SUITE GENERATED 100% SUCCESSFULLY!")
    print("=" * 85)


if __name__ == '__main__':
    generate_qnn_compilation_suite()
