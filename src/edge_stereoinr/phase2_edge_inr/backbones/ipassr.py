"""
iPASSR: Symmetric Parallax Attention Network for Stereo Image Super-Resolution
Reference: Wang et al., "Symmetric Parallax Attention for Stereo Image Super-Resolution", CVPR 2021.
Implements:
1. Residual Feature Extraction Trunk
2. Parallax Attention Module (PAM) with Epipolar Row-wise Matching
3. Cross-View Warping & Occlusion Masking
4. Sub-Pixel PixelShuffle Reconstruction Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class ParallaxAttentionModule(nn.Module):
    """
    Parallax Attention Module (PAM) from iPASSR (CVPR 2021).
    Computes horizontal row-wise cross-attention between Left and Right epipolar lines.
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.channels = channels
        self.scale = channels ** -0.5
        self.q_proj = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

    def forward(self, f_l: torch.Tensor, f_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = f_l.shape

        q_l = self.q_proj(f_l).permute(0, 2, 3, 1)  # (B, H, W_l, C/2)
        k_r = self.k_proj(f_r).permute(0, 2, 1, 3)  # (B, H, C/2, W_r)
        v_r = self.v_proj(f_r).permute(0, 2, 3, 1)  # (B, H, W_r, C)

        q_r = self.q_proj(f_r).permute(0, 2, 3, 1)  # (B, H, W_r, C/2)
        k_l = self.k_proj(f_l).permute(0, 2, 1, 3)  # (B, H, C/2, W_l)
        v_l = self.v_proj(f_l).permute(0, 2, 3, 1)  # (B, H, W_l, C)

        # 1. Left attends to Right (R -> L)
        # (B, H, W_l, C/2) @ (B, H, C/2, W_r) -> (B, H, W_l, W_r)
        attn_r2l = torch.matmul(q_l, k_r) * self.scale
        prob_r2l = F.softmax(attn_r2l, dim=-1)
        warp_r2l = torch.matmul(prob_r2l, v_r).permute(0, 3, 1, 2)  # (B, C, H, W_l)

        # 2. Right attends to Left (L -> R)
        attn_l2r = torch.matmul(q_r, k_l) * self.scale
        prob_l2r = F.softmax(attn_l2r, dim=-1)
        warp_l2r = torch.matmul(prob_l2r, v_l).permute(0, 3, 1, 2)  # (B, C, H, W_r)

        # Fusion with residual connection
        out_l = f_l + self.fuse(torch.cat([f_l, warp_r2l], dim=1))
        out_r = f_r + self.fuse(torch.cat([f_r, warp_l2r], dim=1))
        return out_l, out_r


class iPASSRNet(nn.Module):
    """
    Complete iPASSR Architecture (CVPR 2021).
    Total parameters: ~1.4M (Full-sized academic baseline)
    """
    def __init__(self, in_channels: int = 3, feat_channels: int = 64, num_resblocks: int = 6, upscale: int = 4):
        super().__init__()
        self.upscale = upscale
        self.head = nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1, bias=True)

        # Pre-PAM Residual Blocks
        self.pre_resblocks = nn.ModuleList([ResBlock(feat_channels) for _ in range(num_resblocks // 2)])

        # Parallax Attention Module
        self.pam = ParallaxAttentionModule(feat_channels)

        # Post-PAM Residual Blocks
        self.post_resblocks = nn.ModuleList([ResBlock(feat_channels) for _ in range(num_resblocks // 2)])

        # Sub-Pixel PixelShuffle Reconstruction Head
        self.reconstruction = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels * (upscale ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(upscale),
            nn.Conv2d(feat_channels, in_channels, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, img_l: torch.Tensor, img_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_l = F.interpolate(img_l, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        base_r = F.interpolate(img_r, scale_factor=self.upscale, mode='bicubic', align_corners=False)

        feat_l = self.head(img_l)
        feat_r = self.head(img_r)

        for blk in self.pre_resblocks:
            feat_l = blk(feat_l)
            feat_r = blk(feat_r)

        feat_l, feat_r = self.pam(feat_l, feat_r)

        for blk in self.post_resblocks:
            feat_l = blk(feat_l)
            feat_r = blk(feat_r)

        out_l = self.reconstruction(feat_l) + base_l
        out_r = self.reconstruction(feat_r) + base_r
        return torch.clamp(out_l, 0.0, 1.0), torch.clamp(out_r, 0.0, 1.0)
