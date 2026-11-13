"""
Launcher for Improved Scale-Liberated StereoINR Training (Milestone 2)
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.train_scale_liberated_distill_quest3s import run_improved_training

if __name__ == '__main__':
    run_improved_training(epochs=8, crop_size=128, crops_per_pair=2)
