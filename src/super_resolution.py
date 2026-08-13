"""
super_resolution.py

Super-resolution head: PixelShuffle (sub-pixel convolution) upsamples
the denoised feature maps to the target resolution, followed by
FiLM-conditioned NAFBlocks that refine fine structural detail. Output
is a RESIDUAL (not the final image) -- src/model.py adds it on top of
a cheap bicubic baseline (the global residual connection).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.blocks import NAFBlock


class SuperResolutionHead(nn.Module):
    def __init__(self, width: int = 48, scale: int = 2, num_refine_blocks: int = 4, cond_dim: int = 64):
        super().__init__()
        self.scale = scale
        self.pre_upsample = nn.Conv2d(width, width * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.refine_blocks = nn.ModuleList(
            [NAFBlock(width, cond_dim=cond_dim) for _ in range(num_refine_blocks)]
        )
        self.to_residual = nn.Conv2d(width, 1, kernel_size=3, padding=1)

    def forward(self, feat: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            feat: (B, width, H, W) denoised features from the trunk.
            cond: (B, cond_dim) conditioning vector -- scales how much
                detail synthesis happens based on the estimated
                degradation severity (2x vs 4x-equivalent information
                loss), rather than a fixed assumption.
        Returns:
            (B, 1, H*scale, W*scale) learned residual correction.
        """
        x = self.pre_upsample(feat)
        x = self.pixel_shuffle(x)
        for block in self.refine_blocks:
            x = block(x, cond)
        return self.to_residual(x)