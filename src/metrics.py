"""
metrics.py

Evaluation metrics for PRISM-Net: PSNR, SSIM, LPIPS, inference time.

Conventions (documented per master prompt Section 11):
  - Expected image range: [0, 1] (matches GT). Predictions are
    clamped to [0, 1] ONLY for metric computation -- this does not
    modify the model's raw output used elsewhere.
  - Grayscale handling: SSIM/PSNR operate on single-channel arrays
    directly (scikit-image supports this natively). LPIPS requires
    3-channel input, so single-channel images are repeated across
    3 channels before being rescaled to LPIPS's expected [-1, 1] range.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

try:
    import lpips as lpips_pkg
except ImportError:  # pragma: no cover
    lpips_pkg = None

_LPIPS_MODEL_CACHE: dict = {}


def get_lpips_model(net: str = "alex", device: str = "cpu"):
    """Lazily construct (and cache) an LPIPS model. Requires the
    `lpips` package (in requirements.txt)."""
    if lpips_pkg is None:
        raise ImportError("The 'lpips' package is required for compute_lpips(). "
                           "Install it with: pip install lpips")
    key = (net, device)
    if key not in _LPIPS_MODEL_CACHE:
        model = lpips_pkg.LPIPS(net=net).to(device)
        model.eval()
        _LPIPS_MODEL_CACHE[key] = model
    return _LPIPS_MODEL_CACHE[key]


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """pred, target: (1, H, W) or (H, W) tensors, single image."""
    pred_np = np.clip(pred.detach().cpu().numpy().squeeze(), 0.0, data_range)
    target_np = target.detach().cpu().numpy().squeeze()
    return float(sk_psnr(target_np, pred_np, data_range=data_range))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    pred_np = np.clip(pred.detach().cpu().numpy().squeeze(), 0.0, data_range)
    target_np = target.detach().cpu().numpy().squeeze()
    return float(sk_ssim(target_np, pred_np, data_range=data_range))


def compute_lpips(
    pred: torch.Tensor, target: torch.Tensor, device: str = "cpu", net: str = "alex"
) -> float:
    """pred, target: (1, H, W) or (1, 1, H, W) tensors in [0,1]."""
    model = get_lpips_model(net=net, device=device)

    def _prep(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = torch.clamp(x, 0.0, 1.0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x * 2.0 - 1.0).to(device)

    with torch.no_grad():
        d = model(_prep(pred), _prep(target))
    return float(d.mean().item())


class InferenceTimer:
    """Context manager for wall-clock timing, CUDA-synchronized when
    running on GPU so the measurement isn't a lie about async kernels."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.elapsed: Optional[float] = None

    def __enter__(self):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start
        return False


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor, device: str = "cpu") -> dict:
    """Compute PSNR/SSIM/LPIPS for a single (1,H,W) or (H,W) pred/target
    pair. Returns a plain dict of floats -- used by evaluate.py and
    train.py's validation loop."""
    return {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
        "lpips": compute_lpips(pred, target, device=device),
    }