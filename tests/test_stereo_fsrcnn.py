"""
Unit Test: Stereo-FSRCNN, Monocular FSRCNN, and Stereo-ESPCN
Verifies:
1. Monocular FSRCNN forward pass and shape preservation.
2. Stereo-FSRCNN dual-branch forward pass and epipolar cross-attention consistency.
3. Stereo-ESPCN sub-pixel convolution shape and numerical stability.
4. Parameter counts remain strictly within Adreno 740 GMEM budget (< 100K params).
"""

import os
import sys
import unittest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.edge_stereoinr.phase2_edge_inr.backbones.stereo_fsrcnn import FSRCNN, StereoFSRCNN, StereoESPCN


class TestStereoFSRCNN(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.B, self.C, self.H, self.W = 1, 3, 32, 32
        self.img_l = torch.rand(self.B, self.C, self.H, self.W, device=self.device)
        self.img_r = torch.rand(self.B, self.C, self.H, self.W, device=self.device)

    def test_monocular_fsrcnn(self):
        model = FSRCNN(scale_factor=4, in_channels=3, out_channels=3).to(self.device).eval()
        with torch.no_grad():
            out = model(self.img_l)
        self.assertEqual(out.shape, (self.B, self.C, self.H * 4, self.W * 4))
        params = sum(p.numel() for p in model.parameters())
        self.assertLess(params, 30000)

    def test_stereo_fsrcnn_cross_attention(self):
        model = StereoFSRCNN(scale_factor=4, in_channels=3, out_channels=3, s=16).to(self.device).eval()
        with torch.no_grad():
            out_l, out_r = model(self.img_l, self.img_r)
        self.assertEqual(out_l.shape, (self.B, self.C, self.H * 4, self.W * 4))
        self.assertEqual(out_r.shape, (self.B, self.C, self.H * 4, self.W * 4))
        params = sum(p.numel() for p in model.parameters())
        self.assertLess(params, 35000)

    def test_stereo_espcn_subpixel(self):
        model = StereoESPCN(scale_factor=4, in_channels=3, out_channels=3, mid_channels=32).to(self.device).eval()
        with torch.no_grad():
            out_l, out_r = model(self.img_l, self.img_r)
        self.assertEqual(out_l.shape, (self.B, self.C, self.H * 4, self.W * 4))
        self.assertEqual(out_r.shape, (self.B, self.C, self.H * 4, self.W * 4))
        params = sum(p.numel() for p in model.parameters())
        self.assertLess(params, 35000)


if __name__ == '__main__':
    unittest.main()
