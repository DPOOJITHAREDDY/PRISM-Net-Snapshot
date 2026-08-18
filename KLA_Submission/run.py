#!/usr/bin/env python
"""
PRISM-Net submission runner.

Required usage:
    python run.py <input-dir> <output-dir>

The runner:
- loads models/best_model.pth automatically
- processes every .npy file in the input directory
- preserves input filenames
- produces 256x256 float32 grayscale .npy outputs
- creates the output directory if necessary
- uses CUDA when available
- requires no internet connection
- requires no API key
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PRISM-Net model implementation
# ============================================================

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()

        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)

        x = (x - mu) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1)
        x = x + self.bias.view(1, -1, 1, 1)

        return x.to(orig_dtype)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()

        self.proj = nn.Linear(cond_dim, num_features * 2)

        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(
            self.proj.weight,
            mean=0.0,
            std=0.01,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:

        gamma_beta = self.proj(cond)

        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return x * (1.0 + gamma) + beta


class SoftRangeClamp(nn.Module):
    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        init_softness: float = 0.2,
    ):
        super().__init__()

        self.low = low
        self.high = high

        self.raw_softness = nn.Parameter(
            torch.tensor(float(init_softness))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        softness = (
            F.softplus(self.raw_softness) + 1e-3
        )

        above = x > self.high
        below = x < self.low

        y = x

        y = torch.where(
            above,
            self.high
            + softness * torch.tanh(
                (x - self.high) / softness
            ),
            y,
        )

        y = torch.where(
            below,
            self.low
            - softness * torch.tanh(
                (self.low - x) / softness
            ),
            y,
        )

        return y


class NAFBlock(nn.Module):
    def __init__(
        self,
        c: int,
        cond_dim: int | None = None,
        dw_expand: int = 2,
        ffn_expand: int = 2,
    ):
        super().__init__()

        dw_channel = c * dw_expand

        self.conv1 = nn.Conv2d(
            c,
            dw_channel,
            kernel_size=1,
        )

        self.conv2 = nn.Conv2d(
            dw_channel,
            dw_channel,
            kernel_size=3,
            padding=1,
            groups=dw_channel,
        )

        self.sg1 = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                dw_channel // 2,
                dw_channel // 2,
                kernel_size=1,
            ),
        )

        self.conv3 = nn.Conv2d(
            dw_channel // 2,
            c,
            kernel_size=1,
        )

        ffn_channel = c * ffn_expand

        self.conv4 = nn.Conv2d(
            c,
            ffn_channel,
            kernel_size=1,
        )

        self.sg2 = SimpleGate()

        self.conv5 = nn.Conv2d(
            ffn_channel // 2,
            c,
            kernel_size=1,
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(
            torch.full(
                (1, c, 1, 1),
                1e-2,
            )
        )

        self.gamma = nn.Parameter(
            torch.full(
                (1, c, 1, 1),
                1e-2,
            )
        )

        self.cond_dim = cond_dim

        if cond_dim is not None:
            self.film1 = FiLM(cond_dim, c)
            self.film2 = FiLM(cond_dim, c)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:

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


class DegradationEstimationHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        cond_dim: int = 64,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(
                base_channels,
                base_channels * 2,
                3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.to_scalars = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                base_channels * 4,
                base_channels * 2,
            ),

            nn.LeakyReLU(0.1, inplace=True),

            nn.Linear(
                base_channels * 2,
                2,
            ),
        )

        self.embed = nn.Sequential(
            nn.Linear(2, cond_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, x: torch.Tensor):

        feats = self.features(x)

        raw_scalars = self.to_scalars(feats)

        noise_level = torch.sigmoid(
            raw_scalars[:, 0:1]
        )

        severity = torch.sigmoid(
            raw_scalars[:, 1:2]
        )

        descriptors = torch.cat(
            [noise_level, severity],
            dim=1,
        )

        cond = self.embed(descriptors)

        return cond, descriptors


class DualDomainDenoisingTrunk(nn.Module):
    def __init__(
        self,
        width: int = 48,
        num_blocks: int = 8,
        cond_dim: int = 64,
    ):
        super().__init__()

        self.soft_clamp = SoftRangeClamp()

        self.stem = nn.Conv2d(
            2,
            width,
            kernel_size=3,
            padding=1,
        )

        self.blocks = nn.ModuleList(
            [
                NAFBlock(
                    width,
                    cond_dim=cond_dim,
                )
                for _ in range(num_blocks)
            ]
        )

        self.body_out = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x_raw: torch.Tensor,
        cond: torch.Tensor | None = None,
    ):

        raw_stabilized = self.soft_clamp(x_raw)

        safe_arg = torch.clamp(
            1.0 + x_raw,
            min=1e-3,
        )

        log_domain = torch.log(safe_arg)

        dual = torch.cat(
            [raw_stabilized, log_domain],
            dim=1,
        )

        feat = self.stem(dual)

        for block in self.blocks:
            feat = block(feat, cond)

        feat = self.body_out(feat)

        return feat


class SuperResolutionHead(nn.Module):
    def __init__(
        self,
        width: int = 48,
        scale: int = 2,
        num_refine_blocks: int = 4,
        cond_dim: int = 64,
    ):
        super().__init__()

        self.scale = scale

        self.pre_upsample = nn.Conv2d(
            width,
            width * (scale ** 2),
            kernel_size=3,
            padding=1,
        )

        self.pixel_shuffle = nn.PixelShuffle(scale)

        self.refine_blocks = nn.ModuleList(
            [
                NAFBlock(
                    width,
                    cond_dim=cond_dim,
                )
                for _ in range(num_refine_blocks)
            ]
        )

        self.to_residual = nn.Conv2d(
            width,
            1,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        feat: torch.Tensor,
        cond: torch.Tensor | None = None,
    ):

        x = self.pre_upsample(feat)

        x = self.pixel_shuffle(x)

        for block in self.refine_blocks:
            x = block(x, cond)

        return self.to_residual(x)


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

        self.degradation_head = (
            DegradationEstimationHead(
                in_channels=1,
                cond_dim=cond_dim,
            )
        )

        self.trunk = (
            DualDomainDenoisingTrunk(
                width=width,
                num_blocks=num_denoise_blocks,
                cond_dim=cond_dim,
            )
        )

        self.sr_head = (
            SuperResolutionHead(
                width=width,
                scale=scale,
                num_refine_blocks=num_refine_blocks,
                cond_dim=cond_dim,
            )
        )

    def forward(self, x_raw: torch.Tensor):

        cond, _ = self.degradation_head(x_raw)

        feat = self.trunk(
            x_raw,
            cond,
        )

        residual = self.sr_head(
            feat,
            cond,
        )

        baseline = F.interpolate(
            x_raw,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )

        output = baseline + residual

        return output


# ============================================================
# Input / output handling
# ============================================================

def load_input(path: Path) -> torch.Tensor:
    """
    Load a single input .npy file.

    Accepted forms:
        (128, 128)
        (1, 128, 128)

    Returns:
        (1, 1, 128, 128)
    """

    try:
        arr = np.load(path)
    except Exception as exc:
        raise ValueError(
            f"could not read {path.name}: {exc}"
        ) from exc

    arr = np.asarray(arr)

    if arr.shape == (128, 128):
        pass

    elif arr.shape == (1, 128, 128):
        arr = arr[0]

    else:
        raise ValueError(
            f"{path.name}: expected shape "
            f"(128,128) or (1,128,128), "
            f"got {arr.shape}"
        )

    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(
            f"{path.name}: input must contain numeric values"
        )

    arr = arr.astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(arr).all():
        raise ValueError(
            f"{path.name}: input contains NaN or Inf"
        )

    tensor = torch.from_numpy(arr)

    return tensor.unsqueeze(0).unsqueeze(0)


def load_model(device: torch.device) -> PRISMNet:

    submission_dir = Path(__file__).resolve().parent

    weights_path = (
        submission_dir
        / "models"
        / "best_model.pth"
    )

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"model checkpoint not found: {weights_path}"
        )

    # The original PRISM-Net checkpoint contains:
    #   args
    #   model_state
    #
    # The architecture used for the final checkpoint is:
    #   width = 48
    #   denoising blocks = 8
    #   refinement blocks = 4
    #   conditioning dimension = 64

    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False,
    )

    model = PRISMNet(
        width=48,
        num_denoise_blocks=8,
        num_refine_blocks=4,
        cond_dim=64,
        scale=2,
    )

    if "model_state" not in checkpoint:
        raise RuntimeError(
            "checkpoint does not contain 'model_state'"
        )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# Main submission runner
# ============================================================

def main() -> int:

    if len(sys.argv) != 3:
        print(
            "Usage: python run.py <input-dir> <output-dir>",
            file=sys.stderr,
        )
        return 2

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.is_dir():
        print(
            f"ERROR: input directory does not exist: {input_dir}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_files = sorted(
        input_dir.glob("*.npy")
    )

    if not input_files:
        print(
            f"ERROR: no .npy files found in {input_dir}",
            file=sys.stderr,
        )
        return 1

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"PRISM-Net inference"
    )
    print(
        f"Input directory : {input_dir}"
    )
    print(
        f"Output directory: {output_dir}"
    )
    print(
        f"Device          : {device}"
    )
    print(
        f"Input files     : {len(input_files)}"
    )

    model = load_model(device)

    print("Model loaded successfully.")

    processed = 0
    failed = 0

    total_start = time.perf_counter()

    with torch.no_grad():

        for index, input_path in enumerate(input_files, start=1):

            try:

                noisy = load_input(
                    input_path
                ).to(device)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                prediction = model(noisy)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                prediction = prediction.squeeze(
                    0
                ).squeeze(0)

                if prediction.shape != (256, 256):
                    raise ValueError(
                        f"unexpected output shape "
                        f"{tuple(prediction.shape)}, "
                        f"expected (256,256)"
                    )

                prediction = torch.nan_to_num(
                    prediction,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )

                # Submission requirement:
                # output values must remain in [0,1].
                prediction = torch.clamp(
                    prediction,
                    0.0,
                    1.0,
                )

                output = (
                    prediction
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                if output.shape != (256, 256):
                    raise ValueError(
                        f"invalid saved output shape: "
                        f"{output.shape}"
                    )

                if not np.isfinite(output).all():
                    raise ValueError(
                        "output contains NaN or Inf"
                    )

                if output.min() < 0.0 or output.max() > 1.0:
                    raise ValueError(
                        "output contains values outside [0,1]"
                    )

                output_path = (
                    output_dir
                    / input_path.name
                )

                np.save(
                    output_path,
                    output,
                )

                processed += 1

                if (
                    index == 1
                    or index % 25 == 0
                    or index == len(input_files)
                ):
                    print(
                        f"Processed "
                        f"{index}/{len(input_files)}"
                    )

            except Exception as exc:

                failed += 1

                print(
                    f"FAILED: {input_path.name}: {exc}",
                    file=sys.stderr,
                )

    total_time = (
        time.perf_counter()
        - total_start
    )

    print()
    print(
        f"Completed: {processed}/{len(input_files)}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Total inference time: "
        f"{total_time:.3f} seconds"
    )

    if failed:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())