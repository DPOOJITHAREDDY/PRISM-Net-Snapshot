"""
denoising_trunk.py

Dual-domain denoising trunk: processes the raw-domain intensity and
its log(1+x) transform as complementary representations (speckle
noise is multiplicative in the raw domain, approximately additive in
the log domain -- see src/preprocessing.py), through a stack of
FiLM-conditioned NAFBlocks. The raw-domain branch first passes through
a learnable SoftRangeClamp so speckle overshoot pixels are stabilized
without being hard-clipped away.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.blocks import NAFBlock, SoftRangeClamp
from src.preprocessing import log_domain_transform


class DualDomainDenoisingTrunk(nn.Module):
    def __init__(self, width: int = 48, num_blocks: int = 8, cond_dim: int = 64):
        super().__init__()
        self.soft_clamp = SoftRangeClamp()
        self.stem = nn.Conv2d(2, width, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [NAFBlock(width, cond_dim=cond_dim) for _ in range(num_blocks)]
        )
        self.body_out = nn.Conv2d(width, width, kernel_size=3, padding=1)

    def forward(self, x_raw: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_raw: (B, 1, H, W) raw-domain NoisyLR input, UNCLIPPED.
            cond: (B, cond_dim) conditioning vector from the
                degradation estimation head.
        Returns:
            (B, width, H, W) denoised feature maps, still at input
            resolution -- passed to the super-resolution head.
        """
        raw_stabilized = self.soft_clamp(x_raw)
        # log domain is computed from the ORIGINAL raw input (not the
        # soft-clamped version) so the log branch retains full
        # information about the actual measured intensity.
        log_domain = log_domain_transform(x_raw)
        dual = torch.cat([raw_stabilized, log_domain], dim=1)  # (B, 2, H, W)

        feat = self.stem(dual)
        for block in self.blocks:
            feat = block(feat, cond)
        feat = self.body_out(feat)
        return feat