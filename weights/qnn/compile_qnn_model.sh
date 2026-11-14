#!/bin/bash
# Qualcomm QNN SDK Automated Cross-Compilation Script
# Requires: QNN_SDK_ROOT (e.g., /opt/qcom/qnn-v2.20) and ANDROID_NDK_ROOT

set -e
echo "================================================================="
echo "Compiling StudentStereoINR for Snapdragon XR2 Gen 2 Hexagon NPU"
echo "================================================================="

# Step 1: Convert ONNX to QNN C++ source and binary graph
qnn-onnx-converter \
    --input_network weights/onnx/student_stereoinr_fp32.onnx \
    --output_path weights/qnn/student_stereoinr.cpp \
    --input_dim img_left 1,3,128,128 \
    --input_dim img_right 1,3,128,128 \
    --act_quant_param_method MSE \
    --param_quantizer_method MSE

# Step 2: Compile Model Shared Library (.so) for aarch64-android
qnn-model-lib-generator \
    -c weights/qnn/student_stereoinr.cpp \
    -b weights/qnn/student_stereoinr.bin \
    -o weights/qnn/libstudent_stereoinr.so \
    -t aarch64-android

# Step 3: Generate Hexagon HTP Context Binary (Serialized NPU Cache)
# This enables sub-millisecond cold start without runtime JIT compilation!
qnn-context-binary-generator \
    --backend libQnnHtp.so \
    --model weights/qnn/libstudent_stereoinr.so \
    --binary_file weights/qnn/student_stereoinr_htp.bin \
    --htp_soc_id 43 \
    --log_level info

echo "[SUCCESS] Context binary generated: weights/qnn/student_stereoinr_htp.bin"
echo "Ready for zero-latency execution on Meta Quest 3S Hexagon NPU!"
