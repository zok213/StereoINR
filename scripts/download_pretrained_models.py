"""
Pretrained model download utility for StereoINR and related baselines.
"""

import argparse
import os
from os import path as osp

def download_file(file_id, save_path):
    try:
        import gdown
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    gdown.download(id=file_id, output=save_path, quiet=False)


def download_pretrained_models(method, file_ids):
    save_path_root = f'./experiments/pretrained_models/{method}'
    os.makedirs(save_path_root, exist_ok=True)

    for file_name, file_id in file_ids.items():
        save_path = osp.abspath(osp.join(save_path_root, file_name))
        if osp.exists(save_path):
            print(f"[Skip] {file_name} already exists at {save_path}")
        else:
            print(f"[Download] Fetching {file_name} to {save_path}...")
            download_file(file_id, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download Pretrained Model Checkpoints")
    parser.add_argument(
        'method',
        type=str,
        default='StereoINR',
        nargs='?',
        help="Options: 'StereoINR', 'ESRGAN', 'EDVR', 'StyleGAN', 'EDSR', 'DUF', 'BasicVSR', or 'all'.")
    args = parser.parse_args()

    file_ids = {
        'StereoINR': {
            'StereoINR.pth': '1nkqxDjfKJYl2yyO7r9cEO_IMBlAAJto7'
        },
        'EDSR': {
            'EDSR_Mx2_f64b16_DIV2K_official-3ba7b086.pth': '1mREMGVDymId3NzIc2u90sl_X4-pb4ZcV',
            'EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth': '1bCK6cFYU01uJudLgUUe-jgx-tZ3ikOWn',
            'EDSR_Lx2_f256b32_DIV2K_official-be38e77d.pth': '15257lZCRZ0V6F9LzTyZFYbbPrqNjKyMU',
            'EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth': '1GCi30YYCzgMCcgheGWGusP9aWKOAy5vl'
        },
        'ESRGAN': {
            'ESRGAN_SRx4_DF2KOST_official-ff704c30.pth': '1b3_bWZTjNO3iL2js1yWkJfjZykcQgvzT',
            'ESRGAN_PSNR_SRx4_DF2K_official-150ff491.pth': '1swaV5iBMFfg-DL6ZyiARztbhutDCWXMM'
        }
    }

    if args.method == 'all':
        for m in file_ids.keys():
            download_pretrained_models(m, file_ids[m])
    elif args.method in file_ids:
        download_pretrained_models(args.method, file_ids[args.method])
    else:
        print(f"Unknown method '{args.method}'. Available: {list(file_ids.keys())}")
