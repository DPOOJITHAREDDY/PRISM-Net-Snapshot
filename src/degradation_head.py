"""
degradation_head.py

Lightweight CNN that inspects the raw degraded input and predicts two
degradation descriptors -- an estimated noise level and an estimated
downsample/degradation severity -- then embeds them into a
higher-dimensional conditioning vector consumed by FiLM layers
throughout the denoising trunk and SR head (src/denoising_trunk.py,
src/super_resolution.py).

This estimate is produced at inference time for ANY input, including
out-of-distribution samples never seen during training, so the rest
of the network adapts its behavior per-image instead of applying one
fixed transform to every image (Design Principle #4,
"degradation-aware conditioning").
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DegradationEstimationHead(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16, cond_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),      # 128 -> 64
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),  # 64 -> 32
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),  # 32 -> 16
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_scalars = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(base_channels * 2, 2),  # [noise_level_logit, severity_logit]
        )
        self.embed = nn.Sequential(
            nn.Linear(2, cond_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, H, W) raw-domain NoisyLR input.
        Returns:
            cond: (B, cond_dim) conditioning vector for FiLM layers.
            descriptors: (B, 2) the two estimated scalars in [0,1]
                (noise_level, degradation_severity) -- exposed
                separately for ablation logging / documentation, not
                just consumed internally.
        """
        feats = self.features(x)
        raw_scalars = self.to_scalars(feats)
        noise_level = torch.sigmoid(raw_scalars[:, 0:1])
        severity = torch.sigmoid(raw_scalars[:, 1:2])
        descriptors = torch.cat([noise_level, severity], dim=1)
        cond = self.embed(descriptors)
        return cond, descriptors