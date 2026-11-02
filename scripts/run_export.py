#!/usr/bin/env python3
"""
Master ONNX Export Runner for Edge-StereoINR (Meta Quest 3S)
Exports trained TextureStereoPixelNet to ONNX FP32 and FP16 for Unity Sentis & Native Vulkan.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main():
    print("=" * 85)
    print(">>> EDGE-STEREOINR: MASTER ONNX EXPORT RUNNER")
    print("=" * 85)

    from scripts.deployment.export_texture_pixelnet_onnx import export_texture_model
    export_texture_model()


if __name__ == '__main__':
    main()
