"""
Continuous Arbitrary-Scale RepDyT Stereo INR for Mobile XR (Meta Quest 3S)
Combines:
1. Structural Reparameterization (RepDyT): Multi-branch training dynamics collapsed to fused 3x3 conv.
2. Continuous Arbitrary-Scale Modulation (DyTScaleAdapter) for continuous zoom in [1.0, 4.0].
3. Windowed Stereo Cross-Attention (WindowedSCAM) [-2, 14] for SRAM-bounded epipolar fusion.
4. Fast 4x PixelShuffle path + TMU-friendly continuous grid resampling.
5. Strict Meta Quest 3S Hardware Budget: 117K params (230 KB FP16 weights), 100% GMEM SRAM resident.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_nafrssr import WindowedSCAM, DynamicTanh
from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_continuous_scale_inr import DyTScaleAdapter
from src.edge_stereoinr.phase2_edge_inr.backbones.rep_dyt_block import RepDyTNAFBlock


class ContinuousRepDyTEdgeINR(nn.Module):
    """
    Structural Reparameterized Continuous Stereo INR (RepDyT-Edge-INR)
    Production architecture for Meta Quest 3S (Adreno 740 Vulkan Compute).
    """
    def __init__(self, in_channels: int = 3, width: int = 48, num_blks: int = 4, min_disp: int = -2, max_disp: int = 14):
        super().__init__()
        self.width = width
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)
        self.stereo_fusion = WindowedSCAM(width, min_disp=min_disp, max_disp=max_disp)
        self.blocks = nn.ModuleList([RepDyTNAFBlock(width) for _ in range(num_blks)])
        self.scale_adapter = DyTScaleAdapter(width)

        # 4x Integer Sub-Pixel Path
        self.up_4x = nn.Sequential(
            nn.Conv2d(width, in_channels * 16, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(4)
        )
        nn.init.zeros_(self.up_4x[0].weight)
        nn.init.zeros_(self.up_4x[0].bias)

        # Continuous Arbitrary-Scale Head
        self.cont_head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=True),
            DynamicTanh(width),
            nn.Conv2d(width, in_channels, kernel_size=1, bias=True)
        )
        nn.init.zeros_(self.cont_head[2].weight)
        nn.init.zeros_(self.cont_head[2].bias)

    def switch_to_deploy(self):
        """Collapses multi-branch training convolutions into single-branch inference convolutions."""
        for blk in self.blocks:
            blk.switch_to_deploy()

    def extract_stereo_features(self, img_l: torch.Tensor, img_r: torch.Tensor, scale: float = 4.0):
        fl = self.intro(img_l)
        fr = self.intro(img_r)

        # Base cross-attention
        fl, fr = self.stereo_fusion(fl, fr)

        # Interleaved Recurrent Blocks
        for i, blk in enumerate(self.blocks):
            if i > 0:
                fl, fr = self.stereo_fusion(fl, fr)
            fl = blk(fl)
            fr = blk(fr)

        fl = self.scale_adapter(fl, scale)
        fr = self.scale_adapter(fr, scale)
        return fl, fr

    def forward(self, img_l: torch.Tensor, img_r: torch.Tensor, scale: float = 4.0):
        fl, fr = self.extract_stereo_features(img_l, img_r, scale=scale)

        if abs(scale - 4.0) < 1e-3:
            bl = F.interpolate(img_l, scale_factor=4.0, mode='bicubic', align_corners=False)
            br = F.interpolate(img_r, scale_factor=4.0, mode='bicubic', align_corners=False)
            out_l = self.up_4x(fl) + bl
            out_r = self.up_4x(fr) + br
        else:
            B, C, H, W = img_l.shape
            out_h = int(round(H * scale))
            out_w = int(round(W * scale))
            bl = F.interpolate(img_l, size=(out_h, out_w), mode='bicubic', align_corners=False)
            br = F.interpolate(img_r, size=(out_h, out_w), mode='bicubic', align_corners=False)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, out_h, device=img_l.device),
                torch.linspace(-1, 1, out_w, device=img_l.device),
                indexing='ij'
            )
            grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).repeat(B, 1, 1, 1)

            res_l = F.grid_sample(fl, grid, mode='bilinear', padding_mode='border', align_corners=False)
            res_r = F.grid_sample(fr, grid, mode='bilinear', padding_mode='border', align_corners=False)

            out_l = self.cont_head(res_l) + bl
            out_r = self.cont_head(res_r) + br

        return torch.clamp(out_l, 0.0, 1.0), torch.clamp(out_r, 0.0, 1.0)
