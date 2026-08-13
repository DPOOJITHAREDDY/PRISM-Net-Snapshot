"""
blocks.py

Shared building blocks for PRISM-Net:
  - LayerNorm2d: channel-wise layer norm for (B,C,H,W) feature maps
  - SimpleGate: the gated-linear-unit-style nonlinearity used by NAFNet
  - FiLM: Feature-wise Linear Modulation
  - SoftRangeClamp: learnable soft clipping for speckle overshoot
  - NAFBlock: attention-free restoration block
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """LayerNorm computed over the channel dimension of a (B,C,H,W) tensor.
    
    Computed in float32 internally to avoid numerical instability in fp16
    (variance computation is fragile under half precision). Input and output
    dtypes are preserved.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve original dtype and compute in fp32 for stability.
        orig_dtype = x.dtype
        x = x.float()
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x.to(orig_dtype)


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies them."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: y = x * (1 + gamma) + beta."""

    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, num_features * 2)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.proj(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class SoftRangeClamp(nn.Module):
    """Differentiable, learnable alternative to hard clipping."""

    def __init__(self, low: float = 0.0, high: float = 1.0, init_softness: float = 0.2):
        super().__init__()
        self.low = low
        self.high = high
        self.raw_softness = nn.Parameter(torch.tensor(float(init_softness)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        softness = torch.nn.functional.softplus(self.raw_softness) + 1e-3
        above = x > self.high
        below = x < self.low
        y = x
        y = torch.where(above, self.high + softness * torch.tanh((x - self.high) / softness), y)
        y = torch.where(below, self.low - softness * torch.tanh((self.low - x) / softness), y)
        return y


class NAFBlock(nn.Module):
    """Attention-free restoration block: 1x1 -> depthwise 3x3 -> simplified
    channel attention -> SimpleGate -> 1x1, wrapped with LayerNorm2d and
    a learnable residual scale. Optionally FiLM-conditioned."""

    def __init__(self, c: int, cond_dim: Optional[int] = None, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel)
        self.sg1 = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1),
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1)

        ffn_channel = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # Small nonzero init (not exact zero) to preserve gradient flow.
        self.beta = nn.Parameter(torch.full((1, c, 1, 1), 1e-2))
        self.gamma = nn.Parameter(torch.full((1, c, 1, 1), 1e-2))

        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.film1 = FiLM(cond_dim, c)
            self.film2 = FiLM(cond_dim, c)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = x
        x = self.norm1(x)
        if self.cond_dim is not None and cond is not None:
            x = self.film1(x, cond)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = y + x * self.beta

        x = self.norm2(y)
        if self.cond_dim is not None and cond is not None:
            x = self.film2(x, cond)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        y = y + x * self.gamma
        return y