#!/usr/bin/env python3
"""
Master Benchmark Runner for Edge-StereoINR (Meta Quest 3S)
Executes comprehensive evaluation:
1. Near-field vergence disparity anchoring verification (35 cm reading distance).
2. TextureStereoPixelNet vs Bicubic baseline (PSNR, Laplacian Energy, HFSR, OCR Character Error Rate).
3. Mobile GPU Adreno 740 Tile GMEM residency and latency analysis.
"""

import os
import sys
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Master Benchmark Runner for Meta Quest 3S Passthrough SR")
    parser.add_argument('--suite', type=str, default='all', choices=['all', 'texture', 'vergence', 'adreno'],
                        help="Benchmark suite to run (default: all)")
    args = parser.parse_args()

    print("=" * 85)
    print(">>> EDGE-STEREOINR: MASTER BENCHMARK RUNNER (META QUEST 3S)")
    print("=" * 85)

    if args.suite in ['all', 'vergence']:
        print("\n--- [SUITE 1/3] VERIFYING NEAR-FIELD VERGENCE DISPARITY ANCHORING ---")
        from scripts.benchmarks.verify_nearfield_vergence_anchoring import run_vergence_experiment
        run_vergence_experiment()

    if args.suite in ['all', 'texture']:
        print("\n--- [SUITE 2/3] EVALUATING TEXTURE PRESERVATION & OCR STROKE RECOVERY ---")
        from scripts.benchmarks.benchmark_texture_and_ocr_recovery import run_texture_benchmark
        run_texture_benchmark()

    if args.suite in ['all', 'adreno']:
        print("\n--- [SUITE 3/3] EVALUATING ADRENO 740 GMEM RESIDENCY & SUB-PIXEL CONV ---")
        from scripts.benchmarks.benchmark_inr_vs_pixelshuffle_adreno import run_empirical_benchmark
        run_empirical_benchmark()

    print("\n" + "=" * 85)
    print(">>> ALL BENCHMARK SUITES COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == '__main__':
    main()
