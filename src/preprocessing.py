"""
preprocessing.py

Numerically-safe preprocessing utilities shared between the dataset
loader, the model, and inference.

Handles two things called out explicitly in the problem materials:
  1. NoisyLR values are NOT guaranteed to lie in [0, 1] -- speckle
     noise can push pixels above 1.0 or slightly below 0.0 (observed
     range in the provided dataset: approximately -0.04995 to 1.68157).
     This must NOT be blindly clipped away.
  2. The log-domain representation y = log(1 + x) requires (1 + x) > 0.
     Given the observed minimum (~-0.05), 1 + x stays comfortably
     positive, but OOD test data is not guaranteed to respect that
     bound, so the log transform still needs a numerically-safe floor.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Small positive floor so log1p never receives a non-positive argument,
# even on unseen OOD inputs whose minimum value we cannot guarantee.
LOG_DOMAIN_EPS = 1e-3


def to_chw_tensor(np_array) -> torch.Tensor:
    """Convert a (H, W) float32 numpy array into a (1, H, W) float tensor."""
    tensor = torch.from_numpy(np_array).float()
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def log_domain_transform(x: torch.Tensor, eps: float = LOG_DOMAIN_EPS) -> torch.Tensor:
    """
    Numerically stable y = log(1 + x).

    Speckle noise multiplies with the signal in the raw domain; taking
    log(1+x) turns that multiplicative noise into an approximately
    additive one, which is easier for a CNN to separate from signal.

    Safety: log1p is only well-defined for x > -1. The observed
    NoisyLR range in this dataset (~-0.04995 to 1.68157) never gets
    close to -1, but unseen OOD test data is not assumed to respect
    that bound. We clamp the *argument* to log1p at a small positive
    floor (`eps`), not x itself -- this keeps the raw-domain branch
    (see `build_dual_domain_input`) completely untouched and only
    protects the log branch from a non-positive argument.
    """
    safe_arg = torch.clamp(1.0 + x, min=eps)
    return torch.log(safe_arg)


def build_dual_domain_input(x: torch.Tensor) -> torch.Tensor:
    """
    Build the 2-channel dual-domain input for the denoising trunk:
    channel 0 = untouched raw-domain intensity, channel 1 = log-domain
    representation. Concatenated on the channel dim of a (B,1,H,W) or
    (1,H,W) tensor -> (B,2,H,W) or (2,H,W).
    """
    log_domain = log_domain_transform(x)
    return torch.cat([x, log_domain], dim=-3)


def bicubic_upsample(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """
    Cheap, non-learned bicubic upsample used as the base for the global
    residual connection. The network only has to learn the correction
    on top of this, not the whole image from scratch. Accepts (B,C,H,W)
    or (C,H,W); returns the same rank.
    """
    squeeze_back = x.dim() == 3
    if squeeze_back:
        x = x.unsqueeze(0)
    out = F.interpolate(x, scale_factor=scale_factor, mode="bicubic", align_corners=False)
    return out.squeeze(0) if squeeze_back else out


def bicubic_resize_to(x: torch.Tensor, size) -> torch.Tensor:
    """Bicubic-resize to an exact (H, W) target (not just a scale factor)."""
    squeeze_back = x.dim() == 3
    if squeeze_back:
        x = x.unsqueeze(0)
    out = F.interpolate(x, size=size, mode="bicubic", align_corners=False)
    return out.squeeze(0) if squeeze_back else out