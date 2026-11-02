#!/usr/bin/env python3
"""
Master Training Runner for Edge-StereoINR (Meta Quest 3S)
Executes training on real stereo pairs (Middlebury + Flickr1024) with TexturePreservingLoss.
"""

import os
import sys
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Master Training Runner for Texture-Centric Stereo SR")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size (default: 4)")
    args = parser.parse_args()

    print("=" * 85)
    print(">>> EDGE-STEREOINR: MASTER TRAINING RUNNER")
    print("=" * 85)

    from scripts.training.train_texture_centric_sr import train_texture_model
    train_texture_model(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
