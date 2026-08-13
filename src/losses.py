"""
losses.py

Composite loss function for PRISM-Net training:

    L = 1.0 * L_Charbonnier + 0.3 * L_MS-SSIM + 0.15 * L_Sobel + 0.1 * L_FFT

Deliberately NO adversarial (GAN) term -- Design Principle #1,
"fidelity over hallucination." GAN losses reward convincing-looking
texture regardless of whether it's real, which is unacceptable when
the output feeds a defect-verification process.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


class CharbonnierLoss(nn.Module):
    """Smooth L1 variant: sqrt((pred-target)^2 + eps^2). Robust
    pixel-level fidelity, less sensitive to rare speckle-outlier
    pixels than plain L1/L2."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps ** 2)
        return loss.mean()


class MSSSIMLoss(nn.Module):
    """1 - MS-SSIM. Aligns training directly with the SSIM evaluation
    metric across multiple scales. Inputs are clamped to [0,1] only
    for this loss term (data_range=1.0 matches the documented GT
    range) -- this clamp is local to the loss and does not affect the
    model's actual output or gradients elsewhere."""

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_c = torch.clamp(pred, 0.0, 1.0)
        target_c = torch.clamp(target, 0.0, 1.0)
        score = ms_ssim(pred_c, target_c, data_range=self.data_range, size_average=True)
        return 1.0 - score


class SobelLoss(nn.Module):
    """L1 distance between Sobel gradients of pred and target. Restores
    edge sharpness without the ringing artifacts naive sharpening
    filters introduce."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _gradients(self, x: torch.Tensor):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pgx, pgy = self._gradients(pred)
        tgx, tgy = self._gradients(target)
        return 0.5 * (F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy))


class FFTLoss(nn.Module):
    """L1 distance in the frequency domain (rFFT2 magnitude/phase via
    complex difference). Explicitly penalizes lost high-frequency
    content, aiding fine-detail recovery in the SR stage."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        return (pred_fft - target_fft).abs().mean()


class CompositeLoss(nn.Module):
    def __init__(
        self,
        w_charbonnier: float = 1.0,
        w_msssim: float = 0.3,
        w_sobel: float = 0.15,
        w_fft: float = 0.1,
    ):
        super().__init__()
        self.w_charbonnier = w_charbonnier
        self.w_msssim = w_msssim
        self.w_sobel = w_sobel
        self.w_fft = w_fft
        self.charbonnier = CharbonnierLoss()
        self.msssim = MSSSIMLoss()
        self.sobel = SobelLoss()
        self.fft = FFTLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l_char = self.charbonnier(pred, target)
        l_ssim = self.msssim(pred, target)
        l_sobel = self.sobel(pred, target)
        l_fft = self.fft(pred, target)

        total = (
            self.w_charbonnier * l_char
            + self.w_msssim * l_ssim
            + self.w_sobel * l_sobel
            + self.w_fft * l_fft
        )
        components = {
            "charbonnier": l_char.item(),
            "ms_ssim_loss": l_ssim.item(),
            "sobel": l_sobel.item(),
            "fft": l_fft.item(),
            "total": total.item(),
        }
        return total, components