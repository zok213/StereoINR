"""
Texture-Preserving & Stroke-Enhancement Loss Functions for Stereo Text Super-Resolution
Designed for Video Passthrough reading on Meta Quest 3S (Snapdragon XR2 Gen 2).

Mathematical Formulations:
1. Wavelet High-Frequency Subband Loss:
   Decomposes image via 2D Haar Wavelet into (LL, LH, HL, HH).
   Supervises LH (vertical strokes), HL (horizontal strokes), and HH (corners/serifs)
   to eliminate the "plastic/blurred text" artifact of pure L1/L2 loss.
2. Gradient Directional Loss (Sobel Stroke Contour Loss):
   Preserves sharp edge transitions and prevents stroke width dilation.
3. Laplacian Texture Energy Penalty:
   Maintains high-frequency spectral energy variance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class HaarWavelet2D(nn.Module):
    """
    Differentiable 2D Haar Wavelet Transform.
    Splits image into 4 subbands:
    - LL: Low-frequency approximation
    - LH: Horizontal high-frequency (captures vertical text strokes)
    - HL: Vertical high-frequency (captures horizontal text strokes)
    - HH: Diagonal high-frequency (captures serifs, stroke junctions, dots)
    """
    def __init__(self):
        super().__init__()
        # 2x2 Haar decomposition filters (normalized by 0.5)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        # Shape: (4, 1, 2, 2)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            (LL, LH, HL, HH) each of shape (B, C, H/2, W/2)
        """
        B, C, H, W = x.shape
        # Expand filters for grouped convolution across RGB channels
        weight = self.filters.repeat(C, 1, 1, 1)  # (4*C, 1, 2, 2)
        subbands = F.conv2d(x, weight, stride=2, groups=C)  # (B, 4*C, H/2, W/2)
        
        subbands = subbands.view(B, C, 4, H // 2, W // 2)
        ll = subbands[:, :, 0, :, :]
        lh = subbands[:, :, 1, :, :]
        hl = subbands[:, :, 2, :, :]
        hh = subbands[:, :, 3, :, :]
        return ll, lh, hl, hh


class SobelGradientLoss(nn.Module):
    """
    Evaluates gradient magnitude and directional fidelity.
    Ensures text stroke edges remain razor-sharp without feathering.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) in [0, 1]
            target: (B, C, H, W) in [0, 1]
        """
        # Convert to grayscale luminance for edge computation
        luma_p = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        luma_t = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

        p_padded = F.pad(luma_p, (1, 1, 1, 1), mode='replicate')
        t_padded = F.pad(luma_t, (1, 1, 1, 1), mode='replicate')

        gx_p = F.conv2d(p_padded, self.sobel_x)
        gy_p = F.conv2d(p_padded, self.sobel_y)
        gx_t = F.conv2d(t_padded, self.sobel_x)
        gy_t = F.conv2d(t_padded, self.sobel_y)

        # L1 difference on gradient components
        loss_grad_x = F.l1_loss(gx_p, gx_t)
        loss_grad_y = F.l1_loss(gy_p, gy_t)
        return loss_grad_x + loss_grad_y


class TexturePreservingLoss(nn.Module):
    """
    Composite Loss for High-Frequency Texture & Text Stroke Super-Resolution:
    L_total = L_charbonnier + lambda_wavelet * L_wavelet + lambda_grad * L_grad + lambda_energy * L_energy
    """
    def __init__(
        self,
        lambda_charb: float = 1.0,
        lambda_wavelet: float = 0.4,
        lambda_grad: float = 0.3,
        lambda_energy: float = 0.1
    ):
        super().__init__()
        self.lambda_charb = lambda_charb
        self.lambda_wavelet = lambda_wavelet
        self.lambda_grad = lambda_grad
        self.lambda_energy = lambda_energy

        self.wavelet = HaarWavelet2D()
        self.grad_loss = SobelGradientLoss()

    def charbonnier_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + eps) - math.sqrt(eps))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Base Charbonnier reconstruction loss
        loss_charb = self.charbonnier_loss(pred, target)

        # 2. Wavelet High-Frequency Subband Loss
        ll_p, lh_p, hl_p, hh_p = self.wavelet(pred)
        ll_t, lh_t, hl_t, hh_t = self.wavelet(target)

        # Weight higher on diagonal serifs (HH) and stroke borders (LH, HL)
        loss_lh = F.l1_loss(lh_p, lh_t)
        loss_hl = F.l1_loss(hl_p, hl_t)
        loss_hh = F.l1_loss(hh_p, hh_t)
        loss_wavelet = loss_lh + loss_hl + 1.5 * loss_hh

        # 3. Sobel Edge Gradient Loss
        loss_grad = self.grad_loss(pred, target)

        # 4. High-Frequency Spectral Energy Loss (Penalizes over-smoothing)
        # Ratio of high-frequency variance in prediction vs target
        var_pred_hh = torch.var(hh_p)
        var_target_hh = torch.var(hh_t)
        loss_energy = F.l1_loss(var_pred_hh, var_target_hh)

        # Total Loss
        total_loss = (
            self.lambda_charb * loss_charb +
            self.lambda_wavelet * loss_wavelet +
            self.lambda_grad * loss_grad +
            self.lambda_energy * loss_energy
        )

        return {
            'total_loss': total_loss,
            'loss_charb': loss_charb,
            'loss_wavelet': loss_wavelet,
            'loss_grad': loss_grad,
            'loss_energy': loss_energy
        }


def test_texture_loss():
    print("=" * 80)
    print(">>> VERIFYING TEXTURE-PRESERVING & STROKE-ENHANCEMENT LOSS FUNCTIONS")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Compute Device: {device}")

    loss_fn = TexturePreservingLoss().to(device)

    # 1. Simulate a Ground Truth sharp text image (256x256)
    B, C, H, W = 1, 3, 256, 256
    gt = torch.zeros((B, C, H, W), device=device)
    # Add crisp high-frequency character strokes (font simulation)
    gt[:, :, 50:180, 40:60] = 1.0   # Vertical stroke
    gt[:, :, 50:70, 40:140] = 1.0   # Top horizontal stroke
    gt[:, :, 110:130, 40:120] = 1.0 # Mid horizontal stroke

    # 2. Simulate blurred reconstruction (e.g. standard bicubic or L1 output)
    blurred = F.avg_pool2d(gt, kernel_size=5, stride=1, padding=2)

    # 3. Compute loss
    losses = loss_fn(blurred, gt)

    print(f"[OK] Loss Computation Successful!")
    print(f"     - Total Loss:     {losses['total_loss'].item():.5f}")
    print(f"     - Charbonnier:    {losses['loss_charb'].item():.5f}")
    print(f"     - Wavelet Subband:{losses['loss_wavelet'].item():.5f} (Penalizes blurred strokes)")
    print(f"     - Sobel Gradient: {losses['loss_grad'].item():.5f} (Penalizes soft edges)")
    print(f"     - Spectral Energy:{losses['loss_energy'].item():.5f} (Prevents smoothing collapse)")

    # 4. Verify Identity (Loss on perfect prediction must be near zero)
    zero_losses = loss_fn(gt, gt)
    print(f"[OK] Perfect Match Total Loss: {zero_losses['total_loss'].item():.6f} (Correctly ~0)")
    assert zero_losses['total_loss'].item() < 1e-4

    print("=" * 80)


if __name__ == '__main__':
    test_texture_loss()
