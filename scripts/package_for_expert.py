#!/usr/bin/env python3
"""
Automated Package Generator for Expert & Peer Review
Bundles the clean, essential Edge-StereoINR production framework into a lightweight,
professional zip file (< 3 MB) ready to send to professors, principal engineers, or reviewers.

Excludes:
- Heavy datasets (900+ images)
- 354 MB raw academic baseline weights (includes the 1 MB trained model and 334 KB FP16 ONNX)
- Legacy archives and scratch directories
- Temporary caches and bytecode
"""

import os
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def create_expert_package(output_zip: str = "EdgeStereoINR_Quest3S_ExpertReviewPackage.zip"):
    print("=" * 85)
    print(">>> PACKAGING EDGE-STEREOINR FOR EXPERT REVIEW")
    print("=" * 85)

    output_path = ROOT_DIR / output_zip
    if output_path.exists():
        output_path.unlink()

    included_directories = [
        "configs",
        "docs",
        "scripts",
        "src",
    ]

    included_files = [
        "README.md",
        ".gitignore",
    ]

    # Specific lightweight production weights to include
    production_weights = [
        "weights/checkpoints/texture_stereopixelnet_trained.pth",
        "weights/onnx/texture_stereopixelnet_fp16.onnx",
        "weights/onnx/texture_stereopixelnet_fp32.onnx",
        "weights/onnx/va_stereopixelnet_fp16.onnx",
    ]

    total_files_packed = 0

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # 1. Add top-level individual files
        for f_name in included_files:
            fp = ROOT_DIR / f_name
            if fp.exists():
                zipf.write(fp, arcname=f_name)
                total_files_packed += 1
                print(f"  [+] Added: {f_name}")

        # 2. Add essential directories
        for d_name in included_directories:
            dp = ROOT_DIR / d_name
            if dp.exists():
                for root, dirs, files in os.walk(dp):
                    # Skip __pycache__
                    if "__pycache__" in root:
                        continue
                    for file in files:
                        if file.endswith(('.pyc', '.pyo', '.tmp')):
                            continue
                        full_fp = Path(root) / file
                        rel_path = full_fp.relative_to(ROOT_DIR)
                        zipf.write(full_fp, arcname=str(rel_path))
                        total_files_packed += 1

                print(f"  [+] Added directory: {d_name}/")

        # 3. Add core production weights
        for w_rel in production_weights:
            wp = ROOT_DIR / w_rel
            if wp.exists():
                zipf.write(wp, arcname=w_rel)
                total_files_packed += 1
                print(f"  [+] Added production weight: {w_rel}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 85)
    print(f"[OK] EXPERT REVIEW PACKAGE CREATED SUCCESSFULLY!")
    print(f"     File Name:        {output_path.name}")
    print(f"     Full Path:        {output_path.resolve()}")
    print(f"     Total Files:      {total_files_packed} files (Clean & Minimal!)")
    print(f"     Package Size:     {file_size_mb:.2f} MB (Ready to email or share!)")
    print("=" * 85)


if __name__ == '__main__':
    create_expert_package()
