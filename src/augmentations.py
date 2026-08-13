"""
augmentations.py

Data augmentation for PRISM-Net training.

Two independent pieces, composed together:

1. PairedGeometricAugment — grayscale-safe geometric augmentation
   (random crop, horizontal/vertical flip, 90-degree rotation) applied
   IDENTICALLY to the NoisyLR/GT pair so their spatial correspondence
   is preserved. No color augmentation is used (data is single-channel
   grayscale, so it would be meaningless).

2. CurriculumDegradationAugment — additional synthetic speckle /
   Gaussian / blur degradation layered ON TOP of the already-degraded
   NoisyLR input (never applied to GT), with severity that ramps up
   over training according to a curriculum schedule.

IMPORTANT DESIGN NOTE (documented per the master requirements, Section 9):
The 3,200 training samples are ALREADY paired (degraded_input -> clean_GT).
We do not re-degrade GT or replace the existing NoisyLR degradation --
we only optionally ADD further randomized degradation on top of the
existing NoisyLR image, keeping GT untouched. This keeps
"degraded input -> clean GT" a valid supervised pair while giving the
model exposure to a wider range of degradation severities than the
fixed severity baked into the provided dataset, directly targeting
OOD generalization (see PRISM-Net_Technical_Report.docx, Section 4.4
and 9.2).

DOMAIN-BALANCED SAMPLING — DOCUMENTED DEVIATION:
The technical report (Section 4.4) proposes "domain-balanced batch
sampling across semiconductor structure types." The ACTUAL verified
dataset (master prompt, Section 3) contains no chip-type / domain
label of any kind -- only NoisyLR/GT.npy pairs keyed by a bare numeric
filename. There is nothing to balance against. Rather than fabricate
domain labels that do not exist, `DomainBalancedSampler` below is
implemented as an explicit, documented no-op (falls back to uniform
random shuffling) and performs real balancing if it is ever pointed at
a dataset that DOES carry a recoverable domain label, so this can be
upgraded without silently staying a no-op if the data changes. This
decision is also recorded in docs/REQUIREMENTS_TRACEABILITY.md
(Phase 16).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

# ---------------------------------------------------------------------------
# 1. Geometric augmentation (paired, grayscale-safe)
# ---------------------------------------------------------------------------


class PairedGeometricAugment:
    """
    Applies an identical random geometric transform to a NoisyLR (1,H,W)
    / GT (1,2H,2W) pair: random patch crop, horizontal flip, vertical
    flip, and a random 90-degree rotation. No color/intensity jitter --
    the data is single-channel grayscale, so that class of augmentation
    does not apply.

    Args:
        noisy_crop_size: patch size to crop from the NoisyLR image
            (e.g. 96 crops a 96x96 patch from the 128x128 input). The
            GT crop is taken at exactly `scale`x the size and offset,
            inferred from the actual GT/NoisyLR size ratio (2x for this
            dataset). Set to None to disable cropping (use full images).
        hflip_p / vflip_p: probability of each flip.
        rotate_p: probability of applying a random 90-degree rotation
            (90/180/270, chosen uniformly).
    """

    def __init__(
        self,
        noisy_crop_size: Optional[int] = 96,
        hflip_p: float = 0.5,
        vflip_p: float = 0.5,
        rotate_p: float = 0.5,
    ):
        self.noisy_crop_size = noisy_crop_size
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotate_p = rotate_p

    def _random_crop(self, noisy: torch.Tensor, gt: torch.Tensor):
        if self.noisy_crop_size is None:
            return noisy, gt

        _, h, w = noisy.shape
        crop = self.noisy_crop_size
        if crop >= h or crop >= w:
            return noisy, gt  # image already <= crop size; skip

        top = random.randint(0, h - crop)
        left = random.randint(0, w - crop)
        noisy_crop = noisy[:, top:top + crop, left:left + crop]

        scale = gt.shape[-1] // w  # GT is `scale`x the NoisyLR resolution
        gt_top, gt_left, gt_crop = top * scale, left * scale, crop * scale
        gt_crop_img = gt[:, gt_top:gt_top + gt_crop, gt_left:gt_left + gt_crop]

        return noisy_crop, gt_crop_img

    def __call__(self, sample: dict) -> dict:
        noisy, gt = sample["noisy"], sample["gt"]

        noisy, gt = self._random_crop(noisy, gt)

        if random.random() < self.hflip_p:
            noisy = torch.flip(noisy, dims=[-1])
            gt = torch.flip(gt, dims=[-1])

        if random.random() < self.vflip_p:
            noisy = torch.flip(noisy, dims=[-2])
            gt = torch.flip(gt, dims=[-2])

        if random.random() < self.rotate_p:
            k = random.choice([1, 2, 3])  # 90 / 180 / 270 degrees
            noisy = torch.rot90(noisy, k=k, dims=[-2, -1])
            gt = torch.rot90(gt, k=k, dims=[-2, -1])

        sample["noisy"] = noisy.contiguous()
        sample["gt"] = gt.contiguous()
        return sample


# ---------------------------------------------------------------------------
# 2. Curriculum-scheduled degradation augmentation
# ---------------------------------------------------------------------------


@dataclass
class CurriculumStage:
    """Severity levels for one point in the curriculum."""
    speckle_severity: float
    gaussian_severity: float
    blur_severity: float
    apply_prob: float  # probability ANY extra degradation is applied at all


class CurriculumSchedule:
    """
    Maps a training progress fraction (epoch / total_epochs, in [0, 1])
    to a CurriculumStage.

    Early training (progress < warmup_frac): light, single-degradation
    exposure only (low severity, applied less often), so the model
    first learns the "easy" version of the problem.

    Later training: severity and application probability ramp up
    linearly to the full combined degradation by the end of training,
    per "early epochs use light single-degradation samples, later
    epochs ramp to the full combined degradation" (technical report,
    Section 4.4).
    """

    def __init__(
        self,
        warmup_frac: float = 0.15,
        max_speckle_severity: float = 0.15,
        max_gaussian_severity: float = 0.15,
        max_blur_severity: float = 0.30,
        max_apply_prob: float = 0.7,
        min_apply_prob: float = 0.15,
    ):
        self.warmup_frac = warmup_frac
        self.max_speckle_severity = max_speckle_severity
        self.max_gaussian_severity = max_gaussian_severity
        self.max_blur_severity = max_blur_severity
        self.max_apply_prob = max_apply_prob
        self.min_apply_prob = min_apply_prob

    def stage_for_progress(self, progress: float) -> CurriculumStage:
        progress = max(0.0, min(1.0, progress))

        if progress < self.warmup_frac:
            # Light warmup: only one degradation type at low severity,
            # applied rarely.
            ramp = progress / max(self.warmup_frac, 1e-8)
            return CurriculumStage(
                speckle_severity=0.15 * self.max_speckle_severity * ramp,
                gaussian_severity=0.0,   # single-degradation only during warmup
                blur_severity=0.0,
                apply_prob=self.min_apply_prob * ramp,
            )

        # Post-warmup: linear ramp from light to full combined degradation.
        post_progress = (progress - self.warmup_frac) / max(1.0 - self.warmup_frac, 1e-8)
        post_progress = max(0.0, min(1.0, post_progress))

        severity_scale = 0.2 + 0.8 * post_progress  # 0.2 -> 1.0
        apply_prob = self.min_apply_prob + (self.max_apply_prob - self.min_apply_prob) * post_progress

        return CurriculumStage(
            speckle_severity=self.max_speckle_severity * severity_scale,
            gaussian_severity=self.max_gaussian_severity * severity_scale,
            blur_severity=self.max_blur_severity * severity_scale,
            apply_prob=apply_prob,
        )


def _apply_speckle(x: torch.Tensor, severity: float) -> torch.Tensor:
    """Multiplicative speckle: x' = x * (1 + n), n ~ N(0, severity).
    Matches the physical model documented for this challenge (speckle
    multiplies with the signal, can push values above the clean range)."""
    if severity <= 0:
        return x
    sigma = severity * random.uniform(0.3, 1.0)
    noise = torch.randn_like(x) * sigma
    return x * (1.0 + noise)


def _apply_gaussian(x: torch.Tensor, severity: float) -> torch.Tensor:
    """Additive Gaussian noise: x' = x + n, n ~ N(0, severity)."""
    if severity <= 0:
        return x
    sigma = severity * random.uniform(0.3, 1.0) * 0.25  # keep additive noise on a sane scale
    noise = torch.randn_like(x) * sigma
    return x + noise


def _apply_blur(x: torch.Tensor, severity: float) -> torch.Tensor:
    """
    Mild extra softening to simulate additional resolution/detail loss,
    implemented as a small separable Gaussian blur kernel rather than
    an actual resize -- the NoisyLR spatial size (128x128) must stay
    fixed to match the model input, so "more resolution lost" is
    approximated with blur instead of literal downsampling.
    """
    if severity <= 0:
        return x
    ksize = 3 if severity < 0.5 else 5
    sigma = 0.3 + severity * 1.2
    coords = torch.arange(ksize, dtype=x.dtype, device=x.device) - ksize // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = (g / g.sum()).view(1, 1, -1)

    x_b = x.unsqueeze(0)  # (1, C, H, W)
    pad = ksize // 2
    x_b = F.pad(x_b, (pad, pad, pad, pad), mode="reflect")
    c = x_b.shape[1]
    kh = kernel_1d.view(1, 1, 1, ksize).repeat(c, 1, 1, 1)
    kv = kernel_1d.view(1, 1, ksize, 1).repeat(c, 1, 1, 1)
    x_b = F.conv2d(x_b, kh, groups=c)
    x_b = F.conv2d(x_b, kv, groups=c)
    return x_b.squeeze(0)


class CurriculumDegradationAugment:
    """
    Applies additional randomized speckle / Gaussian / blur degradation
    to the NoisyLR image ONLY (GT is never touched), with severity
    controlled by a CurriculumSchedule. Call `set_epoch(epoch,
    total_epochs)` once per training epoch (train.py does this, Phase 9)
    to advance the curriculum; `__call__` applies a randomized subset
    and order of degradations at the current severity to each sample.
    """

    def __init__(self, schedule: Optional[CurriculumSchedule] = None):
        self.schedule = schedule or CurriculumSchedule()
        self.stage = self.schedule.stage_for_progress(0.0)

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        progress = epoch / max(total_epochs - 1, 1)
        self.stage = self.schedule.stage_for_progress(progress)

    def __call__(self, sample: dict) -> dict:
        if random.random() >= self.stage.apply_prob:
            return sample  # no extra degradation this sample

        noisy = sample["noisy"]

        # Randomize which subset and order of degradations are applied,
        # per "randomized combinations/order where appropriate".
        ops = []
        if self.stage.speckle_severity > 0 and random.random() < 0.7:
            ops.append(lambda t: _apply_speckle(t, self.stage.speckle_severity))
        if self.stage.gaussian_severity > 0 and random.random() < 0.7:
            ops.append(lambda t: _apply_gaussian(t, self.stage.gaussian_severity))
        if self.stage.blur_severity > 0 and random.random() < 0.5:
            ops.append(lambda t: _apply_blur(t, self.stage.blur_severity))
        random.shuffle(ops)

        for op in ops:
            noisy = op(noisy)

        # Safety bound for synthetic curriculum degradation. This does not
        # clip the original dataset; it only bounds synthetic degradation
        # introduced by the curriculum so extreme random combinations cannot
        # destabilize training.
        noisy = torch.clamp(noisy, min=-0.5, max=2.5)

        sample["noisy"] = noisy
        return sample


# ---------------------------------------------------------------------------
# 3. Composed training transform
# ---------------------------------------------------------------------------


class ComposeTrainTransform:
    """Geometric augmentation, then curriculum degradation augmentation."""

    def __init__(self, geometric: PairedGeometricAugment, degradation: CurriculumDegradationAugment):
        self.geometric = geometric
        self.degradation = degradation

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        self.degradation.set_epoch(epoch, total_epochs)

    def __call__(self, sample: dict) -> dict:
        sample = self.geometric(sample)
        sample = self.degradation(sample)
        return sample


# ---------------------------------------------------------------------------
# 4. Domain-balanced sampling -- documented no-op (see module docstring)
# ---------------------------------------------------------------------------


class DomainBalancedSampler(Sampler):
    """
    Real domain-balanced sampler that falls back to uniform-random
    shuffling when no domain label is available (see module docstring
    for why that's the case for THIS dataset, not a shortcut).

    Args:
        stems: the list of sample filename stems this epoch draws from.
        domain_of: optional `stem -> str` function. If provided and it
            maps stems to more than one distinct value, real per-domain
            round-robin balancing is used. Otherwise this logs a clear
            one-time notice and falls back to shuffling.
    """

    def __init__(self, stems, domain_of=None, seed: int = 42):
        self.stems = list(stems)
        self.seed = seed
        self._epoch = 0

        domains = None
        if domain_of is not None:
            domains = {s: domain_of(s) for s in self.stems}

        self.balanced = domains is not None and len(set(domains.values())) > 1
        self.domains = domains

        if not self.balanced:
            print(
                "[DomainBalancedSampler] No usable domain/structure-type labels "
                "found in this dataset -- falling back to uniform random "
                "shuffling. See src/augmentations.py module docstring and "
                "docs/REQUIREMENTS_TRACEABILITY.md for the documented reason."
            )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        if not self.balanced:
            order = list(range(len(self.stems)))
            rng.shuffle(order)
            return iter(order)

        # Real balanced path (only reachable with a genuine domain_of fn).
        buckets: dict = {}
        for i, s in enumerate(self.stems):
            buckets.setdefault(self.domains[s], []).append(i)
        for idx_list in buckets.values():
            rng.shuffle(idx_list)

        order = []
        pointers = {k: 0 for k in buckets}
        remaining = sum(len(v) for v in buckets.values())
        keys = list(buckets.keys())
        while remaining > 0:
            rng.shuffle(keys)
            for k in keys:
                p = pointers[k]
                if p < len(buckets[k]):
                    order.append(buckets[k][p])
                    pointers[k] += 1
                    remaining -= 1
        return iter(order)

    def __len__(self) -> int:
        return len(self.stems)