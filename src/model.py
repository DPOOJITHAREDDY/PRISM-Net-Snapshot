"""
model.py

PRISM-Net: the full end-to-end model.

    Input (B,1,128,128)
      |
      +--> Degradation Estimation Head --> conditioning vector
      |
      v  (FiLM-conditioned throughout)
    Dual-Domain Denoising Trunk (raw + log domain, soft-range clamp)
      v
    Super-Resolution Head (PixelShuffle + refinement, FiLM-conditioned)
      v
    + bicubic-upsampled input (Global Residual Connection)
      v
    Output (B,1,256,256)

The network only learns the correction on top of the bicubic
baseline, per Design Principle: "the model should primarily learn
the restoration correction, not reconstruct the whole image from
scratch."
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.degradation_head import DegradationEstimationHead
from src.denoising_trunk import DualDomainDenoisingTrunk
from src.preprocessing import bicubic_upsample
from src.super_resolution import SuperResolutionHead


class PRISMNet(nn.Module):
    def __init__(
        self,
        width: int = 48,
        num_denoise_blocks: int = 8,
        num_refine_blocks: int = 4,
        cond_dim: int = 64,
        scale: int = 2,
    ):
        super().__init__()
        self.scale = scale
        self.degradation_head = DegradationEstimationHead(in_channels=1, cond_dim=cond_dim)
        self.trunk = DualDomainDenoisingTrunk(width=width, num_blocks=num_denoise_blocks, cond_dim=cond_dim)
        self.sr_head = SuperResolutionHead(
            width=width, scale=scale, num_refine_blocks=num_refine_blocks, cond_dim=cond_dim
        )

    def forward(self, x_raw: torch.Tensor, return_aux: bool = False):
        """
        Args:
            x_raw: (B, 1, 128, 128) raw NoisyLR input, UNCLIPPED
                (values may lie outside [0, 1]).
            return_aux: if True, also return a dict with the
                conditioning vector, the two degradation descriptors,
                and the bicubic baseline -- useful for ablations and
                debugging, not required for normal inference.
        Returns:
            output: (B, 1, 256, 256) restored image.
            aux (optional): dict, see above.
        """
        cond, descriptors = self.degradation_head(x_raw)
        feat = self.trunk(x_raw, cond)
        residual = self.sr_head(feat, cond)
        baseline = bicubic_upsample(x_raw, scale_factor=self.scale)
        output = baseline + residual

        if return_aux:
            aux = {
                "cond": cond,
                "degradation_descriptors": descriptors,
                "bicubic_baseline": baseline,
                "residual": residual,
            }
            return output, aux
        return output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)