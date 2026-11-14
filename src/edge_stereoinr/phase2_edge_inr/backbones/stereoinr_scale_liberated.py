"""
StereoINR-ScaleLiberated: Improved 2026 Re-Architecture (RepDyT-EdgeINR)
========================================================================
Addresses all Milestone 1 critique bottlenecks (M1 -> M2 Blueprint):
1. RepDyT Structural Reparameterization: Multi-branch (3x3 + 1x1 + Dirac identity)
   during training, collapsing via switch_to_deploy() to fused single 3x3 Conv2d
   with exactly 4,468,180 parameters (0 added parameter or latency overhead).
2. Residual Variance Damping: Scale residual branches by 0.2 (res_scale=0.2) to
   prevent unscaled activation variance expansion (eliminating std explosion to 21.6).
3. Dynamic Tanh (DyT) with Lipschitz Bound: Clamped alpha (Gamma_max <= 2.0)
   guaranteeing zero clipping during Symmetric INT8 Post-Training Quantization (PTQ).
4. Strict 1D Epipolar Parallax Attention (EPA-1D): Scanline-restricted (y_L = y_R)
   preserving stereo disparity consistency and eliminating retinal rivalry.
5. Unified Multi-Task Loss Suite: Charbonnier (eps=1e-3) + Bidirectional Horizontal
   Epipolar Warping with Left-Right Consistency (LRC) Occlusion Masking (tau=1.0)
   + 2D FFT Spectral Loss + Knowledge Distillation (KD) from 43.8M StereoINR.
   (Completely eliminates flawed loss_disp = 0.01 * disp^2).
"""

import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicTanh(nn.Module):
    """
    Hardware-friendly element-wise activation with strict Lipschitz bound enforcement.
    Replacing LayerNorm (zero thread barrier synchronization).
    Guarantees Lipschitz constant Gamma <= max_gamma (default: 2.0) for INT8 PTQ.
    """
    def __init__(self, channels: int, max_gamma: float = 2.0):
        super().__init__()
        self.channels = channels
        self.max_gamma = float(max_gamma)
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @property
    def lipschitz_constant(self) -> float:
        """Returns the maximum empirical Lipschitz bound |alpha_clamped|."""
        with torch.no_grad():
            clamped = torch.clamp(self.alpha, -self.max_gamma, self.max_gamma)
            return float(clamped.abs().max().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_clamped = torch.clamp(self.alpha, -self.max_gamma, self.max_gamma)
        return torch.tanh(x * alpha_clamped) + self.beta


class RepConv2d(nn.Module):
    """
    Structural Reparameterization 2D Convolution:
    - Training mode: multi-branch (3x3 conv + 1x1 conv + Dirac identity).
    - Inference mode: single fused 3x3 conv (0 extra latency or parameters).
    - Dirac warm-start: supports exact function equivalence to pretrained 3x3 weights.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.has_bias = bias
        self.has_identity = (in_channels == out_channels)
        self.deploy = False

        # Multi-branch training components
        self.branch_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=bias)
        self.branch_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)

        # Fused single branch for deployment
        self.fused_conv = None

        self.dirac_warm_start()

    def dirac_warm_start(self, pretrained_3x3: Optional[torch.Tensor] = None):
        """
        Initializes branches. If pretrained_3x3 is supplied, shifts 3x3 by Dirac delta
        so that at initialization: W_3x3 + W_1x1 + W_dirac == W_pretrained.
        """
        device = self.branch_3x3.weight.device
        if pretrained_3x3 is not None:
            w_init = pretrained_3x3.detach().clone().to(device)
            w_dirac = torch.zeros_like(w_init)
            if self.has_identity:
                for i in range(min(self.in_channels, self.out_channels)):
                    w_dirac[i, i, 1, 1] = 1.0
            self.branch_3x3.weight.data.copy_(w_init - w_dirac)
            self.branch_1x1.weight.data.zero_()
            if self.has_bias and self.branch_3x3.bias is not None:
                self.branch_3x3.bias.data.zero_()
                self.branch_1x1.bias.data.zero_()
        else:
            nn.init.kaiming_normal_(self.branch_3x3.weight, mode='fan_out', nonlinearity='relu')
            nn.init.normal_(self.branch_1x1.weight, std=1e-3)
            if self.has_bias and self.branch_3x3.bias is not None:
                nn.init.zeros_(self.branch_3x3.bias)
                nn.init.zeros_(self.branch_1x1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.fused_conv(x)

        out = self.branch_3x3(x) + self.branch_1x1(x)
        if self.has_identity:
            out = out + x
        return out

    def get_fused_weights(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Algebraically fuses multi-branch weights into a single 3x3 kernel."""
        w_3x3 = self.branch_3x3.weight
        b_3x3 = self.branch_3x3.bias if self.has_bias and self.branch_3x3.bias is not None else None

        w_1x1 = self.branch_1x1.weight
        b_1x1 = self.branch_1x1.bias if self.has_bias and self.branch_1x1.bias is not None else None

        w_1x1_padded = F.pad(w_1x1, [1, 1, 1, 1])

        w_identity = torch.zeros_like(w_3x3)
        if self.has_identity:
            for i in range(min(self.in_channels, self.out_channels)):
                w_identity[i, i, 1, 1] = 1.0

        fused_w = w_3x3 + w_1x1_padded + w_identity
        fused_b = None
        if self.has_bias:
            b_3 = b_3x3 if b_3x3 is not None else torch.zeros(self.out_channels, device=w_3x3.device)
            b_1 = b_1x1 if b_1x1 is not None else torch.zeros(self.out_channels, device=w_1x1.device)
            fused_b = b_3 + b_1

        return fused_w, fused_b

    def switch_to_deploy(self):
        """Permanently collapses multi-branch into a single 3x3 conv."""
        if self.deploy:
            return
        fused_w, fused_b = self.get_fused_weights()
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=3, padding=self.padding,
            bias=(fused_b is not None)
        ).to(fused_w.device)

        self.fused_conv.weight.data.copy_(fused_w)
        if fused_b is not None:
            self.fused_conv.bias.data.copy_(fused_b)

        # Release training branches
        self.__delattr__('branch_3x3')
        self.__delattr__('branch_1x1')
        self.deploy = True


class EpipolarAttention(nn.Module):
    """
    1D Epipolar Parallax Attention Block (EPA-1D):
    Restricts cross-view correlation strictly along horizontal scanlines (y_L = y_R).
    Preserves bilateral stereo parallax without 2D deformable sampling.
    """
    def __init__(self, channels: int = 144, heads: int = 4):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5

        self.qkv_l = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_r = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dyt = DynamicTanh(channels, max_gamma=2.0)

    def forward(self, xl: torch.Tensor, xr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = xl.shape
        # Project Q, K, V
        ql, kl, vl = self.qkv_l(xl).view(B, 3, self.heads, self.head_dim, H, W).unbind(1)
        qr, kr, vr = self.qkv_r(xr).view(B, 3, self.heads, self.head_dim, H, W).unbind(1)

        # Scanline restriction: (B, heads, H, W, head_dim) x (B, heads, H, head_dim, W)
        # H is strictly a batch dimension, isolating scanlines
        ql_s = ql.permute(0, 1, 3, 4, 2)
        kr_s = kr.permute(0, 1, 3, 2, 4)
        vr_s = vr.permute(0, 1, 3, 4, 2)

        attn_r2l = torch.softmax(torch.matmul(ql_s, kr_s) * self.scale, dim=-1)
        fused_l = torch.matmul(attn_r2l, vr_s).permute(0, 1, 4, 2, 3).reshape(B, C, H, W)
        out_l = xl + 0.2 * self.dyt(self.proj(fused_l))

        qr_s = qr.permute(0, 1, 3, 4, 2)
        kl_s = kl.permute(0, 1, 3, 2, 4)
        vl_s = vl.permute(0, 1, 3, 4, 2)

        attn_l2r = torch.softmax(torch.matmul(qr_s, kl_s) * self.scale, dim=-1)
        fused_r = torch.matmul(attn_l2r, vl_s).permute(0, 1, 4, 2, 3).reshape(B, C, H, W)
        out_r = xr + 0.2 * self.dyt(self.proj(fused_r))

        return out_l, out_r


class RepDyTResidualBlock(nn.Module):
    """
    Residual Block with RepConv2d and Dynamic Tanh:
    Applies residual scaling (res_scale=0.2) to eliminate activation variance explosion.
    """
    def __init__(self, channels: int = 144, res_scale: float = 0.2):
        super().__init__()
        self.channels = channels
        self.res_scale = float(res_scale)

        self.conv1 = RepConv2d(channels, channels, bias=False)
        self.dyt = DynamicTanh(channels, max_gamma=2.0)
        self.conv2 = RepConv2d(channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.dyt(self.conv1(x)))
        return x + self.res_scale * res

    def switch_to_deploy(self):
        self.conv1.switch_to_deploy()
        self.conv2.switch_to_deploy()


class ScaleLiberatedStereoINR(nn.Module):
    """
    Scale-Liberated Continuous Stereo INR Architecture (DyT-EdgeINR).
    Exact Deploy Parameter Budget: 4,468,180 parameters (8.52 MB FP16 / 4.26 MB INT8).
    
    Architectural Specification:
    - conv_first: Conv2d(3 -> 144) + DynamicTanh(144) [4,176 params]
    - blocks: 8 x RepDyTResidualBlock(144) [2,988,288 params in deploy mode]
    - stereo_adapters: 8 x EpipolarAttention(144) [1,163,520 params]
    - disp_net: Conv2d(288 -> 64) + GELU + Conv2d(64 -> 1) + Tanh [166,529 params]
    - mlp: Linear(308 -> 256) + GELU + Linear(256 -> 256) + GELU + Linear(256 -> 3) [145,667 params]
    - Total Deploy Parameters: 4,468,180.
    """
    def __init__(self, embed_dim: int = 144, num_blocks: int = 8, res_scale: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.res_scale = float(res_scale)
        self.deploy = False

        # Input Stem
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=3, padding=1, bias=False),
            DynamicTanh(embed_dim, max_gamma=2.0)
        )

        # 8 Residual Blocks with RepConv2d + Residual Damping
        self.blocks = nn.ModuleList([
            RepDyTResidualBlock(embed_dim, res_scale=res_scale)
            for _ in range(num_blocks)
        ])

        # 8 Epipolar Parallax Attention Modules
        self.stereo_adapters = nn.ModuleList([
            EpipolarAttention(embed_dim, heads=4)
            for _ in range(num_blocks)
        ])

        # Epipolar Disparity Estimator Network
        self.disp_net = nn.Sequential(
            nn.Conv2d(embed_dim * 2, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # Continuous INR MLP Decoder (in_dim: 144*2 + 18 + 2 = 308)
        in_dim = embed_dim * 2 + 18 + 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 3)
        )
        # Zero-initialize final projection for monotonic residual refinement over Bicubic anchor
        nn.init.zeros_(self.mlp[4].weight)
        nn.init.zeros_(self.mlp[4].bias)

        freqs = [2.0 ** i * math.pi for i in range(4)]
        self.register_buffer('freqs', torch.tensor(freqs, dtype=torch.float32))

    def switch_to_deploy(self):
        """Collapses all RepConv2d branches into single fused 3x3 Conv2d."""
        if self.deploy:
            return
        for b in self.blocks:
            b.switch_to_deploy()
        self.deploy = True

    def warp(self, feat: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """
        Differentiable horizontal epipolar warping strictly along scanlines (y_L = y_R).
        disp is normalized displacement in [-1.0, 1.0].
        """
        B, C, H, W = feat.shape
        x_base = torch.arange(W, dtype=torch.float32, device=feat.device).view(1, 1, 1, W).expand(B, 1, H, W)
        x_src = torch.clamp(x_base + disp * ((W - 1) / 2.0), 0.0, float(W - 1))
        x0 = torch.floor(x_src).long()
        x1 = torch.clamp(x0 + 1, max=W - 1)
        w1 = x_src - x0.float()
        w0 = 1.0 - w1
        return w0 * torch.gather(feat, 3, x0.expand(-1, C, -1, -1)) + w1 * torch.gather(feat, 3, x1.expand(-1, C, -1, -1))

    def forward_features(self, img_l: torch.Tensor, img_r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts student intermediate feature representations and disparity maps."""
        fl = self.conv_first(img_l)
        fr = self.conv_first(img_r)

        for b, a in zip(self.blocks, self.stereo_adapters):
            fl = b(fl)
            fr = b(fr)
            fl, fr = a(fl, fr)

        # Disparity estimation
        cat_lr = torch.cat([fl, fr], dim=1)
        cat_rl = torch.cat([fr, fl], dim=1)
        dl2r = self.disp_net(cat_lr)
        dr2l = self.disp_net(cat_rl)

        return fl, fr, dl2r, dr2l

    def forward(
        self,
        img_l: torch.Tensor,
        img_r: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        scale: float = 4.0,
        out_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Dual-mode forward pass:
        1. coords is not None: Continuous coordinate querying across arbitrary points.
        2. coords is None: Full-resolution continuous super-resolution at arbitrary scale factor.
        """
        B, C, H, W = img_l.shape
        fl, fr, dl2r, dr2l = self.forward_features(img_l, img_r)

        fr_w = self.warp(fr, dl2r)
        fl_w = self.warp(fl, dr2l)

        if coords is not None:
            N = coords.shape[1]
            grid = coords.unsqueeze(1)
            fl_pts = F.grid_sample(fl, grid, mode='bilinear', align_corners=False).squeeze(2).permute(0, 2, 1)
            frw_pts = F.grid_sample(fr_w, grid, mode='bilinear', align_corners=False).squeeze(2).permute(0, 2, 1)
            fr_pts = F.grid_sample(fr, grid, mode='bilinear', align_corners=False).squeeze(2).permute(0, 2, 1)
            flw_pts = F.grid_sample(fl_w, grid, mode='bilinear', align_corners=False).squeeze(2).permute(0, 2, 1)

            base_l = F.grid_sample(img_l, grid, mode='bilinear', align_corners=False).squeeze(2).permute(0, 2, 1)
            base_r = F.grid_sample(img_r, grid, mode='bilinear', align_corners=False).squeeze(2).permute(0, 2, 1)

            pos = [coords]
            for frq in self.freqs:
                pos.extend([torch.sin(coords * frq), torch.cos(coords * frq)])
            pos_enc = torch.cat(pos, dim=-1)

            cell = torch.tensor([2.0 / (H * scale), 2.0 / (W * scale)], device=img_l.device).view(1, 1, 2).expand(B, N, -1)

            inp_l = torch.cat([fl_pts, frw_pts, pos_enc, cell], dim=-1)
            inp_r = torch.cat([fr_pts, flw_pts, pos_enc, cell], dim=-1)

            srl_pts = torch.clamp(base_l + self.mlp(inp_l), 0.0, 1.0)
            srr_pts = torch.clamp(base_r + self.mlp(inp_r), 0.0, 1.0)

            return srl_pts, srr_pts, {'disp_l2r': dl2r, 'disp_r2l': dr2l}

        else:
            # Full continuous regular grid decoding
            out_H, out_W = out_size if out_size else (int(round(H * scale)), int(round(W * scale)))
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1.0 + 1.0 / out_H, 1.0 - 1.0 / out_H, out_H, device=img_l.device),
                torch.linspace(-1.0 + 1.0 / out_W, 1.0 - 1.0 / out_W, out_W, device=img_l.device),
                indexing='ij'
            )
            full_coords = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 2).expand(B, -1, -1)

            # Query continuous INR points in chunks to bound peak memory
            N_total = full_coords.shape[1]
            chunk_size = 32768
            srl_chunks, srr_chunks = [], []

            pos_full = [full_coords]
            for frq in self.freqs:
                pos_full.extend([torch.sin(full_coords * frq), torch.cos(full_coords * frq)])
            pos_enc_full = torch.cat(pos_full, dim=-1)
            cell = torch.tensor([2.0 / out_H, 2.0 / out_W], device=img_l.device).view(1, 1, 2).expand(B, N_total, -1)

            for i in range(0, N_total, chunk_size):
                sub_coords = full_coords[:, i:i+chunk_size, :]
                sub_grid = sub_coords.unsqueeze(1)

                fl_sub = F.grid_sample(fl, sub_grid, mode='bilinear', align_corners=False).squeeze(2).permute(0, 2, 1)
                frw_sub = F.grid_sample(fr_w, sub_grid, mode='bilinear', align_corners=False).squeeze(2).permute(0, 2, 1)
                fr_sub = F.grid_sample(fr, sub_grid, mode='bilinear', align_corners=False).squeeze(2).permute(0, 2, 1)
                flw_sub = F.grid_sample(fl_w, sub_grid, mode='bilinear', align_corners=False).squeeze(2).permute(0, 2, 1)

                base_l_sub = F.grid_sample(img_l, sub_grid, mode='bilinear', align_corners=False).squeeze(2).permute(0, 2, 1)
                base_r_sub = F.grid_sample(img_r, sub_grid, mode='bilinear', align_corners=False).squeeze(2).permute(0, 2, 1)

                inp_l_sub = torch.cat([fl_sub, frw_sub, pos_enc_full[:, i:i+chunk_size, :], cell[:, i:i+chunk_size, :]], dim=-1)
                inp_r_sub = torch.cat([fr_sub, flw_sub, pos_enc_full[:, i:i+chunk_size, :], cell[:, i:i+chunk_size, :]], dim=-1)

                res_l_sub = self.mlp(inp_l_sub)
                res_r_sub = self.mlp(inp_r_sub)

                srl_chunks.append(res_l_sub)
                srr_chunks.append(res_r_sub)

            res_l = torch.cat(srl_chunks, dim=1).view(B, out_H, out_W, 3).permute(0, 3, 1, 2)
            res_r = torch.cat(srr_chunks, dim=1).view(B, out_H, out_W, 3).permute(0, 3, 1, 2)

            bic_l = F.interpolate(img_l, size=(out_H, out_W), mode='bicubic', align_corners=False)
            bic_r = F.interpolate(img_r, size=(out_H, out_W), mode='bicubic', align_corners=False)

            out_l = torch.clamp(bic_l + res_l, 0.0, 1.0)
            out_r = torch.clamp(bic_r + res_r, 0.0, 1.0)

            return out_l, out_r, {'disp_l2r': dl2r, 'disp_r2l': dr2l}

    def load_pretrained_baseline(self, ckpt_path: str):
        """Loads weights from baseline checkpoint, mapping to RepConv Dirac warm-start."""
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']

        # Load conv_first
        if 'conv_first.0.weight' in sd:
            self.conv_first[0].weight.data.copy_(sd['conv_first.0.weight'])
        if 'conv_first.1.alpha' in sd:
            self.conv_first[1].alpha.data.copy_(sd['conv_first.1.alpha'])
        if 'conv_first.1.beta' in sd:
            self.conv_first[1].beta.data.copy_(sd['conv_first.1.beta'])

        # Load blocks with Dirac warm-start
        for i, b in enumerate(self.blocks):
            w1_key = f'blocks.{i}.0.weight'
            a_key = f'blocks.{i}.1.alpha'
            b_key = f'blocks.{i}.1.beta'
            w2_key = f'blocks.{i}.2.weight'

            if w1_key in sd:
                b.conv1.dirac_warm_start(sd[w1_key])
            if a_key in sd:
                b.dyt.alpha.data.copy_(sd[a_key])
            if b_key in sd:
                b.dyt.beta.data.copy_(sd[b_key])
            if w2_key in sd:
                b.conv2.dirac_warm_start(sd[w2_key])

        # Load adapters, disp_net, mlp
        for i, a in enumerate(self.stereo_adapters):
            if f'stereo_adapters.{i}.qkv_l.weight' in sd:
                a.qkv_l.weight.data.copy_(sd[f'stereo_adapters.{i}.qkv_l.weight'])
            if f'stereo_adapters.{i}.qkv_r.weight' in sd:
                a.qkv_r.weight.data.copy_(sd[f'stereo_adapters.{i}.qkv_r.weight'])
            if f'stereo_adapters.{i}.proj.weight' in sd:
                a.proj.weight.data.copy_(sd[f'stereo_adapters.{i}.proj.weight'])
            if f'stereo_adapters.{i}.dyt.alpha' in sd:
                a.dyt.alpha.data.copy_(sd[f'stereo_adapters.{i}.dyt.alpha'])
            if f'stereo_adapters.{i}.dyt.beta' in sd:
                a.dyt.beta.data.copy_(sd[f'stereo_adapters.{i}.dyt.beta'])

        for k in ['disp_net.0.weight', 'disp_net.0.bias', 'disp_net.2.weight', 'disp_net.2.bias']:
            if k in sd:
                mod = self.disp_net[0] if '0' in k else self.disp_net[2]
                attr = 'weight' if 'weight' in k else 'bias'
                getattr(mod, attr).data.copy_(sd[k])

        for k in ['mlp.0.weight', 'mlp.0.bias', 'mlp.2.weight', 'mlp.2.bias', 'mlp.4.weight', 'mlp.4.bias']:
            if k in sd:
                idx = int(k.split('.')[1])
                attr = 'weight' if 'weight' in k else 'bias'
                getattr(self.mlp[idx], attr).data.copy_(sd[k])

        # Fallback to zero-initialize final layer only if not present in sd
        if 'mlp.4.weight' not in sd:
            nn.init.zeros_(self.mlp[4].weight)
        if 'mlp.4.bias' not in sd:
            nn.init.zeros_(self.mlp[4].bias)


# ==============================================================================
# MULTI-TASK LOSS SUITE (Requirement 1)
# ==============================================================================

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (smooth L1 / pseudo-Huber): robust to outliers and sensor noise."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class BidirectionalEpipolarWarpingLoss(nn.Module):
    """
    Bidirectional Horizontal Epipolar Warping Photometric Consistency Loss.
    Strictly along horizontal scanlines (y_L = y_R) with Left-Right Consistency (LRC)
    occlusion gating (tau=1.0 px).
    """
    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def horizontal_warp(self, img: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """Horizontal epipolar warping along horizontal scanlines."""
        B, C, H, W = img.shape
        x_base = torch.arange(W, dtype=torch.float32, device=img.device).view(1, 1, 1, W).expand(B, 1, H, W)
        x_src = torch.clamp(x_base + disp * ((W - 1) / 2.0), 0.0, float(W - 1))
        x0 = torch.floor(x_src).long()
        x1 = torch.clamp(x0 + 1, max=W - 1)
        w1 = x_src - x0.float()
        w0 = 1.0 - w1
        return w0 * torch.gather(img, 3, x0.expand(-1, C, -1, -1)) + w1 * torch.gather(img, 3, x1.expand(-1, C, -1, -1))

    def forward(
        self,
        img_l: torch.Tensor,
        img_r: torch.Tensor,
        disp_l2r: torch.Tensor,
        disp_r2l: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes LRC occlusion-gated bidirectional warping error.
        disp_l2r and disp_r2l are predicted disparity displacement maps.
        """
        B, C, H, W = img_l.shape

        # Warp images
        warped_r2l = self.horizontal_warp(img_r, disp_l2r)
        warped_l2r = self.horizontal_warp(img_l, disp_r2l)

        # Warp right disparity to left view for LRC check
        # In pixel space, displacement is disp * ((W-1)/2.0)
        disp_scale = (W - 1) / 2.0
        disp_l_px = disp_l2r * disp_scale
        disp_r_px = disp_r2l * disp_scale

        warped_disp_r_to_l = self.horizontal_warp(disp_r_px, disp_l2r)
        warped_disp_l_to_r = self.horizontal_warp(disp_l_px, disp_r2l)

        # LRC Occlusion Mask: |d_L + d_R_warped| < tau (opposite displacement directions)
        mask_l = (torch.abs(disp_l_px + warped_disp_r_to_l) < self.tau).float().detach()
        mask_r = (torch.abs(disp_r_px + warped_disp_l_to_r) < self.tau).float().detach()

        # Photometric error
        err_l = torch.abs(img_l - warped_r2l)
        err_r = torch.abs(img_r - warped_l2r)

        loss_l = (err_l * mask_l).sum() / (mask_l.sum() * C + 1e-6)
        loss_r = (err_r * mask_r).sum() / (mask_r.sum() * C + 1e-6)

        return 0.5 * (loss_l + loss_r)


class FFTSpectralLoss(nn.Module):
    """2D Real FFT Spectral Magnitude Loss across channels."""
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fft_pred = torch.fft.rfft2(pred, norm='ortho')
        fft_target = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(torch.abs(fft_pred), torch.abs(fft_target))


class FeatureKDLoss(nn.Module):
    """
    Teacher-Student Feature Knowledge Distillation Loss.
    Aligns student features (C=144) to 1x1 projected teacher features (C=180).
    """
    def __init__(self, student_dim: int = 144, teacher_dim: int = 180):
        super().__init__()
        self.proj = nn.Conv2d(teacher_dim, student_dim, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        # Match spatial resolution if needed
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            teacher_feat = F.interpolate(teacher_feat, size=student_feat.shape[2:], mode='bilinear', align_corners=False)
        proj_teacher = self.proj(teacher_feat)
        return F.mse_loss(student_feat, proj_teacher)


class MultiTaskLossSuite(nn.Module):
    """
    Unified Multi-Task Loss Suite:
    L_total = 1.0 * L_char + 0.5 * L_warp + 0.1 * L_fft + 0.25 * L_KD
    """
    def __init__(
        self,
        lambda_char: float = 1.0,
        lambda_warp: float = 0.5,
        lambda_fft: float = 0.1,
        lambda_kd: float = 0.25,
        eps_char: float = 1e-3,
        tau_lrc: float = 1.0,
        student_dim: int = 144,
        teacher_dim: int = 180
    ):
        super().__init__()
        self.lambda_char = lambda_char
        self.lambda_warp = lambda_warp
        self.lambda_fft = lambda_fft
        self.lambda_kd = lambda_kd

        self.charbonnier = CharbonnierLoss(eps=eps_char)
        self.warping = BidirectionalEpipolarWarpingLoss(tau=tau_lrc)
        self.fft = FFTSpectralLoss()
        self.kd = FeatureKDLoss(student_dim=student_dim, teacher_dim=teacher_dim)

    def forward(
        self,
        pred_l: torch.Tensor,
        pred_r: torch.Tensor,
        gt_l: torch.Tensor,
        gt_r: torch.Tensor,
        disp_l2r: torch.Tensor,
        disp_r2l: torch.Tensor,
        student_feat_l: Optional[torch.Tensor] = None,
        teacher_feat_l: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # 1. Charbonnier
        l_char = 0.5 * (self.charbonnier(pred_l, gt_l) + self.charbonnier(pred_r, gt_r))

        # 2. Epipolar Warping with LRC
        # Downsample or upsample images to match disparity resolution for photometric warp
        H_d, W_d = disp_l2r.shape[2], disp_l2r.shape[3]
        if pred_l.shape[2:] != (H_d, W_d):
            pred_l_warp = F.interpolate(pred_l, size=(H_d, W_d), mode='bilinear', align_corners=False)
            pred_r_warp = F.interpolate(pred_r, size=(H_d, W_d), mode='bilinear', align_corners=False)
        else:
            pred_l_warp = pred_l
            pred_r_warp = pred_r

        l_warp = self.warping(pred_l_warp, pred_r_warp, disp_l2r, disp_r2l)

        # 3. 2D FFT Spectral Loss
        l_fft = 0.5 * (self.fft(pred_l, gt_l) + self.fft(pred_r, gt_r))

        # 4. Feature KD
        l_kd = torch.tensor(0.0, device=pred_l.device)
        if student_feat_l is not None and teacher_feat_l is not None:
            l_kd = self.kd(student_feat_l, teacher_feat_l)

        # Total Unified Loss
        l_total = (
            self.lambda_char * l_char +
            self.lambda_warp * l_warp +
            self.lambda_fft * l_fft +
            self.lambda_kd * l_kd
        )

        return {
            'loss_total': l_total,
            'loss_char': l_char,
            'loss_warp': l_warp,
            'loss_fft': l_fft,
            'loss_kd': l_kd
        }
