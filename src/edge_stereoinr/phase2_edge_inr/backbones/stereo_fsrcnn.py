"""
Stereo-FSRCNN & Monocular FSRCNN Architectures
Rigorous implementations of:
1. Monocular FSRCNN (Chao Dong et al., ECCV 2016)
2. Stereo-FSRCNN (Ultra-Lightweight Dual-Branch FSRCNN with Epipolar Parallax Attention)
3. Stereo-ESPCN (Real-Time Sub-Pixel Convolution Stereo Baseline, Shi et al., CVPR 2016)

Designed for edge XR and benchmark comparisons on Adreno 740 GMEM SRAM (Meta Quest 3S).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpipolarCrossAttention(nn.Module):
    """
    Lightweight 1D Epipolar Cross-Attention along the horizontal scanline.
    Matches features along epipolar lines (same height y, search along width x).
    """
    def __init__(self, channels):
        super(EpipolarCrossAttention, self).__init__()
        self.channels = channels
        self.q_conv = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.k_conv = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = (channels // 2) ** -0.5

    def forward(self, f_left, f_right):
        """
        f_left, f_right: [B, C, H, W]
        Fuses complementary parallax information across left and right views.
        """
        B, C, H, W = f_left.shape
        
        # Reshape to [B*H, C/2, W] for 1D horizontal epipolar attention
        q_l = self.q_conv(f_left).view(B * H, -1, W).transpose(1, 2)  # [B*H, W, C/2]
        k_r = self.k_conv(f_right).view(B * H, -1, W)                  # [B*H, C/2, W]
        v_r = self.v_conv(f_right).view(B * H, -1, W).transpose(1, 2)  # [B*H, W, C]
        
        attn_lr = torch.bmm(q_l, k_r) * self.scale                    # [B*H, W_l, W_r]
        attn_lr = F.softmax(attn_lr, dim=-1)
        fused_l = torch.bmm(attn_lr, v_r).transpose(1, 2).contiguous().view(B, C, H, W)
        out_l = f_left + self.proj(fused_l)

        # Symmetric Right branch: query right, key/value left
        q_r = self.q_conv(f_right).view(B * H, -1, W).transpose(1, 2) # [B*H, W, C/2]
        k_l = self.k_conv(f_left).view(B * H, -1, W)                   # [B*H, C/2, W]
        v_l = self.v_conv(f_left).view(B * H, -1, W).transpose(1, 2)  # [B*H, W, C]
        
        attn_rl = torch.bmm(q_r, k_l) * self.scale                    # [B*H, W_r, W_l]
        attn_rl = F.softmax(attn_rl, dim=-1)
        fused_r = torch.bmm(attn_rl, v_l).transpose(1, 2).contiguous().view(B, C, H, W)
        out_r = f_right + self.proj(fused_r)

        return out_l, out_r


class FSRCNN(nn.Module):
    """
    Standard Single Image Super-Resolution FSRCNN (Dong et al., ECCV 2016).
    Processes monocular images independently.
    Added optional residual learning (bicubic skip) for modern stabilization.
    """
    def __init__(self, scale_factor=4, in_channels=3, out_channels=3, d=56, s=12, m=4, residual_learning=True):
        super(FSRCNN, self).__init__()
        self.scale_factor = scale_factor
        self.residual_learning = residual_learning
        
        # 1. Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )
        
        # 2. Shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )
        
        # 3. Non-linear mapping
        mapping = []
        for _ in range(m):
            mapping.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mapping.append(nn.PReLU(s))
        self.mapping = nn.Sequential(*mapping)
        
        # 4. Expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )
        
        # 5. Deconvolution (Reconstruction)
        self.deconv = nn.ConvTranspose2d(
            d, out_channels, kernel_size=9, stride=scale_factor,
            padding=9 // 2, output_padding=scale_factor - 1
        )
        if residual_learning:
            # Zero-init deconv so initial output is exactly bicubic
            nn.init.zeros_(self.deconv.weight)
            nn.init.zeros_(self.deconv.bias)

    def forward(self, x):
        feat = self.feature_extraction(x)
        feat = self.shrinking(feat)
        feat = self.mapping(feat)
        feat = self.expanding(feat)
        out = self.deconv(feat)
        if self.residual_learning:
            bicubic = F.interpolate(x, scale_factor=float(self.scale_factor), mode='bicubic', align_corners=False)
            out = out + bicubic
        return out


class StereoFSRCNN(nn.Module):
    """
    Stereo-FSRCNN: Ultra-Lightweight Dual-Branch Stereo Super-Resolution CNN.
    Extends the FSRCNN philosophy to Stereo by integrating Epipolar Cross-Attention
    in the low-dimensional shrunk feature space (s=16).
    """
    def __init__(self, scale_factor=4, in_channels=3, out_channels=3, d=56, s=16, m=4, residual_learning=True):
        super(StereoFSRCNN, self).__init__()
        self.scale_factor = scale_factor
        self.residual_learning = residual_learning
        
        # 1. Shared Feature Extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )
        
        # 2. Shared Shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )
        
        # 3. Epipolar Cross-Attention (fuses Left & Right in shrunk feature space)
        self.stereo_fusion = EpipolarCrossAttention(channels=s)
        
        # 4. Non-linear mapping
        mapping = []
        for _ in range(m):
            mapping.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mapping.append(nn.PReLU(s))
        self.mapping = nn.Sequential(*mapping)
        
        # 5. Shared Expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )
        
        # 6. Shared Deconvolution
        self.deconv = nn.ConvTranspose2d(
            d, out_channels, kernel_size=9, stride=scale_factor,
            padding=9 // 2, output_padding=scale_factor - 1
        )
        if residual_learning:
            nn.init.zeros_(self.deconv.weight)
            nn.init.zeros_(self.deconv.bias)

    def forward(self, x_left, x_right):
        """
        x_left, x_right: [B, 3, H, W]
        Returns: sr_left, sr_right [B, 3, H*scale, W*scale]
        """
        # Shared feature extraction
        f_l = self.feature_extraction(x_left)
        f_r = self.feature_extraction(x_right)
        
        # Shrinking
        f_l_shrunk = self.shrinking(f_l)
        f_r_shrunk = self.shrinking(f_r)
        
        # Epipolar Stereo Fusion
        f_l_fused, f_r_fused = self.stereo_fusion(f_l_shrunk, f_r_shrunk)
        
        # Non-linear mapping
        f_l_map = self.mapping(f_l_fused)
        f_r_map = self.mapping(f_r_fused)
        
        # Expanding
        f_l_exp = self.expanding(f_l_map)
        f_r_exp = self.expanding(f_r_map)
        
        # Deconvolution
        sr_l = self.deconv(f_l_exp)
        sr_r = self.deconv(f_r_exp)
        
        if self.residual_learning:
            b_l = F.interpolate(x_left, scale_factor=float(self.scale_factor), mode='bicubic', align_corners=False)
            b_r = F.interpolate(x_right, scale_factor=float(self.scale_factor), mode='bicubic', align_corners=False)
            sr_l = sr_l + b_l
            sr_r = sr_r + b_r
            
        return sr_l, sr_r


class StereoESPCN(nn.Module):
    """
    Stereo-ESPCN: Real-time Sub-Pixel Convolution with Epipolar Stereo Fusion.
    Uses PixelShuffle instead of ConvTranspose2d to avoid checkerboard artifacts.
    """
    def __init__(self, scale_factor=4, in_channels=3, out_channels=3, mid_channels=32, residual_learning=True):
        super(StereoESPCN, self).__init__()
        self.scale_factor = scale_factor
        self.residual_learning = residual_learning
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.cross_attn = EpipolarCrossAttention(channels=mid_channels)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )
        if residual_learning:
            nn.init.zeros_(self.conv3[0].weight)
            nn.init.zeros_(self.conv3[0].bias)

    def forward(self, x_left, x_right):
        f_l = self.conv2(self.conv1(x_left))
        f_r = self.conv2(self.conv1(x_right))
        f_l_fused, f_r_fused = self.cross_attn(f_l, f_r)
        sr_l = self.conv3(f_l_fused)
        sr_r = self.conv3(f_r_fused)
        if self.residual_learning:
            b_l = F.interpolate(x_left, scale_factor=float(self.scale_factor), mode='bicubic', align_corners=False)
            b_r = F.interpolate(x_right, scale_factor=float(self.scale_factor), mode='bicubic', align_corners=False)
            sr_l = sr_l + b_l
            sr_r = sr_r + b_r
        return sr_l, sr_r
