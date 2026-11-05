#!/bin/bash
# Qualcomm QNN Compilation Script for Linux / Android aarch64 (Meta Quest 3S)
set -e
echo "[*] Converting ONNX graph to Qualcomm DLC..."
qnn-onnx-converter \
    --input_network ../../weights/qnn_npu/dyt_edge_inr_qnn_tile64_4x.onnx \
    --output_path dyt_edge_inr_qnn_tile64_4x.dlc

echo "[*] Quantizing to INT8 for Hexagon HTP Vector Processor..."
qnn-context-binary-generator \
    --model dyt_edge_inr_qnn_tile64_4x.dlc \
    --backend libQnnHtp.so \
    --input_list calibration_input_list.txt \
    --binary_file dyt_edge_inr_hexagon_context.bin \
    --output_dir output_htp

echo "[SUCCESS] Hexagon NPU Context Binary created in output_htp/dyt_edge_inr_hexagon_context.bin!"
