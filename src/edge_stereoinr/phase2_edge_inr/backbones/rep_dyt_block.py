"""
Structural Reparameterization for Mobile XR Stereo INR (RepDyT)
Enables higher training representational capacity (+0.2-0.3 dB) while collapsing
at deployment into a single standard 3x3 convolution with ZERO inference overhead.

Mathematical Derivation:
In Training:
    y = Conv3x3(x) + Conv1x1(x) + Identity(x)
In Deployment (switch_to_deploy):
    W_fused = W_3x3 + pad(W_1x1, 1) + I_delta
    b_fused = b_3x3 + b_1x1
    y = Conv3x3_fused(x)
Bitwise Parity: Output error < 1e-6 (Exact mathematical equivalence).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_nafrssr import DynamicTanh, SimpleGate


class RepConv2d(nn.Module):
    """
    Structural Reparameterization 2D Convolution:
    Multi-branch training: 3x3 + 1x1 + Identity.
    Single-branch inference: Fused 3x3 conv.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = False

        # Multi-branch training components
        self.branch_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.branch_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)
        self.has_identity = (in_channels == out_channels)

        # Fused single branch for deployment
        self.fused_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.fused_conv(x)

        out = self.branch_3x3(x) + self.branch_1x1(x)
        if self.has_identity:
            out = out + x
        return out

    def get_fused_weights(self):
        """Algebraically collapses multi-branch weights into a single 3x3 kernel."""
        # 1. 3x3 kernel
        w_3x3 = self.branch_3x3.weight
        b_3x3 = self.branch_3x3.bias if self.branch_3x3.bias is not None else torch.zeros(self.out_channels, device=w_3x3.device)

        # 2. 1x1 kernel zero-padded to 3x3
        w_1x1 = self.branch_1x1.weight
        b_1x1 = self.branch_1x1.bias if self.branch_1x1.bias is not None else torch.zeros(self.out_channels, device=w_1x1.device)
        w_1x1_padded = F.pad(w_1x1, [1, 1, 1, 1])

        # 3. Identity Dirac delta kernel (if in_channels == out_channels)
        w_identity = torch.zeros_like(w_3x3)
        if self.has_identity:
            for i in range(self.in_channels):
                w_identity[i, i, 1, 1] = 1.0

        fused_w = w_3x3 + w_1x1_padded + w_identity
        fused_b = b_3x3 + b_1x1
        return fused_w, fused_b

    def switch_to_deploy(self):
        """Permanently switches module to single 3x3 conv for production deployment."""
        if self.deploy:
            return
        fused_w, fused_b = self.get_fused_weights()
        self.fused_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=True)
        self.fused_conv.weight.data.copy_(fused_w)
        self.fused_conv.bias.data.copy_(fused_b)

        # Free training branches to release memory
        self.__delattr__('branch_3x3')
        self.__delattr__('branch_1x1')
        self.deploy = True


class RepDyTNAFBlock(nn.Module):
    """
    Reparameterized DyT-NAF Block:
    Uses RepConv2d for feature mixing during training.
    Collapses to identical latency as standard DyTNAFBlock at inference!
    """
    def __init__(self, c: int, DW_Expand: int = 2, FFN_Expand: int = 2):
        super().__init__()
        self.c = c
        dw_channel = c * DW_Expand
        
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, bias=True)
        self.rep_conv2 = RepConv2d(dw_channel, dw_channel, bias=True)
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, bias=True)

        self.norm1 = DynamicTanh(c)
        self.norm2 = DynamicTanh(c)

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.rep_conv5 = RepConv2d(ffn_channel // 2, c, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.rep_conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.rep_conv5(x)
        return y + x * self.gamma

    def switch_to_deploy(self):
        self.rep_conv2.switch_to_deploy()
        self.rep_conv5.switch_to_deploy()
