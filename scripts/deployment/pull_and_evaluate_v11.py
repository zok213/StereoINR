"""
Auto-Pull & Benchmark Pipeline for Kaggle Dual-T4 QuestStereoINR Version 11
Monitors Kaggle kernel status, downloads trained weights, and runs canonical evaluation.
"""

import os
import sys
import subprocess
import shutil

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

KERNEL_SLUG = "zok213/stereoinr-turbo-dual-t4"
OUT_DIR = os.path.join(ROOT_DIR, "weights", "kaggle_dual_t4", "v11_output")


def check_and_pull():
    print(f"[*] Querying status for kernel: {KERNEL_SLUG}...")
    res = subprocess.run(["kaggle", "kernels", "status", KERNEL_SLUG], capture_output=True, text=True)
    out = res.stdout.strip()
    print(out)

    if "KernelWorkerStatus.COMPLETE" in out:
        print("[+] Kernel completed! Downloading output artifacts...")
        os.makedirs(OUT_DIR, exist_ok=True)
        dl_res = subprocess.run(["kaggle", "kernels", "output", KERNEL_SLUG, "-p", OUT_DIR], capture_output=True, text=True)
        print(dl_res.stdout)

        files = os.listdir(OUT_DIR)
        print(f"[*] Downloaded artifacts ({len(files)} files): {files}")

        # Deploy ONNX to Unity Sentis
        onnx_fp16 = os.path.join(OUT_DIR, "quest_stereoinr_fast4x_deploy_fp16.onnx")
        sentis_dist = os.path.join(ROOT_DIR, "deployment", "unity_sentis", "dist")
        os.makedirs(sentis_dist, exist_ok=True)
        if os.path.exists(onnx_fp16):
            shutil.copy(onnx_fp16, os.path.join(sentis_dist, "quest_stereoinr_fast4x_deploy_fp16.onnx"))
            print(f"[+] Deployed Native FP16 ONNX to Unity Sentis: {sentis_dist}")

        # Benchmark against test scenes
        deploy_pth = os.path.join(OUT_DIR, "quest_stereoinr_deploy.pth")
        if os.path.exists(deploy_pth):
            print("[*] Running Canonical Benchmark Suite on trained Kaggle weights...")
            bench_script = os.path.join(ROOT_DIR, "scripts", "benchmarking", "benchmark_kaggle_v11_deploy.py")
            subprocess.run([sys.executable, bench_script, deploy_pth])

        return True
    elif "KernelWorkerStatus.RUNNING" in out:
        print("[*] Kernel is still actively computing on Dual Tesla T4 GPUs...")
        return False
    else:
        print(f"[!] Kernel status: {out}")
        return False


if __name__ == '__main__':
    check_and_pull()
