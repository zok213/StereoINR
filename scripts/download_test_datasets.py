"""
StereoINR: Automated Test Datasets Download & Setup Pipeline
Downloads and unpacks the 4 standard benchmark datasets used in the StereoINR paper:
  - Middlebury (5 test pairs)
  - Flickr1024 (112 test pairs)
  - KITTI 2012 (20 test pairs)
  - KITTI 2015 (20 test pairs)
"""

import os
import sys
import zipfile
import shutil
import argparse

# Google Drive File ID for official test sets archive (~490 MB)
TEST_SETS_GDRIVE_ID = "1LQDUclNtNZWTT41NndISLGvjvuBbxeUs"


def download_test_datasets(target_dir: str = "datasets/StereoSR"):
    test_dir = os.path.join(target_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    expected_datasets = ["Middlebury", "Flickr1024", "KITTI2012", "KITTI2015"]
    all_exist = all(os.path.exists(os.path.join(test_dir, ds, "hr")) for ds in expected_datasets)

    if all_exist:
        print("[Dataset] All 4 benchmark datasets already exist:")
        for ds in expected_datasets:
            hr_path = os.path.join(test_dir, ds, "hr")
            pairs = [d for d in os.listdir(hr_path) if os.path.isdir(os.path.join(hr_path, d))]
            print(f"  - {ds:12}: {len(pairs)} test stereo pairs verified.")
        return

    archive_path = os.path.join(target_dir, "test_sets.zip")
    if not os.path.exists(archive_path):
        print(f"[Dataset] Downloading benchmark test sets archive ({TEST_SETS_GDRIVE_ID})...")
        try:
            import gdown
        except ImportError:
            print("[Dataset] Installing gdown for automated Google Drive download...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

        gdown.download(id=TEST_SETS_GDRIVE_ID, output=archive_path, quiet=False)

    print(f"[Dataset] Extracting archive: {archive_path} -> {target_dir}...")
    with zipfile.ZipFile(archive_path, 'r') as zf:
        zf.extractall(target_dir)

    print("[Dataset] Verification:")
    for ds in expected_datasets:
        hr_path = os.path.join(test_dir, ds, "hr")
        if os.path.exists(hr_path):
            pairs = [d for d in os.listdir(hr_path) if os.path.isdir(os.path.join(hr_path, d))]
            print(f"  - {ds:12}: {len(pairs)} test stereo pairs verified.")
        else:
            print(f"  - {ds:12}: directory missing after extraction!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download StereoINR Test Datasets")
    parser.add_argument("--output_dir", type=str, default="datasets/StereoSR", help="Target output folder")
    args = parser.parse_args()
    download_test_datasets(args.output_dir)
