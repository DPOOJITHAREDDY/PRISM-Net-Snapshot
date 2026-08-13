"""
dataset.py

PyTorch Dataset implementations for PRISM-Net.

- PrismNetTrainDataset: paired (NoisyLR, GT) samples for training/
  validation, matched via src.utils, with an optional transform
  pipeline (augmentation -- src/augmentations.py, Phase 5) applied
  per-sample.
- PrismNetTestDataset: NoisyLR-only samples for inference on the
  official test set (no GT available).

Both:
  - ignore macOS extraction metadata
  - never silently clip NoisyLR values
  - raise a clear error on missing pairs or malformed arrays rather
    than silently skipping them
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from torch.utils.data import Dataset

from src.preprocessing import to_chw_tensor
from src.utils import load_npy_array, scan_paired_dataset, scan_test_dataset


class PrismNetTrainDataset(Dataset):
    """
    Paired NoisyLR/GT dataset for training and validation.

    Args:
        data_root: folder containing NoisyLR/ and GT/ subfolders.
        stems: optional explicit list of filename stems to include
            (used to implement the train/val split below). If None,
            every matched pair found under `data_root` is used.
        transform: optional callable applied to
            {"noisy": tensor, "gt": tensor, "stem": str} -> same dict,
            for augmentation.
        require_no_missing_pairs: if True (default), raise immediately
            if any file lacks its pair.
    """

    def __init__(
        self,
        data_root: str,
        stems: Optional[List[str]] = None,
        transform: Optional[Callable[[dict], dict]] = None,
        require_no_missing_pairs: bool = True,
    ):
        self.data_root = Path(data_root)
        self.transform = transform

        scan = scan_paired_dataset(str(self.data_root))
        if require_no_missing_pairs and (scan.missing_gt or scan.missing_noisy):
            raise ValueError(
                f"Dataset at {self.data_root} has unmatched files: "
                f"{len(scan.missing_gt)} NoisyLR without GT, "
                f"{len(scan.missing_noisy)} GT without NoisyLR. "
                f"Run validate_dataset.py for details."
            )

        self.noisy_files = scan.noisy_files
        self.gt_files = scan.gt_files
        self.stems: List[str] = list(stems) if stems is not None else list(scan.matched)

        missing = [s for s in self.stems if s not in scan.noisy_files or s not in scan.gt_files]
        if missing:
            raise ValueError(
                f"{len(missing)} requested stem(s) missing from NoisyLR/ or GT/, "
                f"e.g. {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        noisy_arr = load_npy_array(self.noisy_files[stem])
        gt_arr = load_npy_array(self.gt_files[stem])

        noisy = to_chw_tensor(noisy_arr)  # (1, 128, 128), NOT clipped
        gt = to_chw_tensor(gt_arr)        # (1, 256, 256), already in [0, 1]

        sample = {"noisy": noisy, "gt": gt, "stem": stem}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class PrismNetTestDataset(Dataset):
    """
    Inference-only dataset over a directory of NoisyLR .npy files with
    no ground truth (the official 400-image test set).

    `input_dir` may point directly at a folder of .npy files, or at a
    parent folder containing a `NoisyLR/` subfolder -- both work, so
    evaluate.py behaves the same whether pointed at
    ".../Test_NoisyLR" or ".../Test_NoisyLR/NoisyLR".
    """

    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.files, self.ignored = scan_test_dataset(str(self.input_dir))
        self.stems: List[str] = sorted(self.files.keys())

        if len(self.stems) == 0:
            raise ValueError(
                f"No valid .npy files found under {self.input_dir} "
                f"(found {len(self.ignored)} ignorable metadata file(s))."
            )

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        noisy_arr = load_npy_array(self.files[stem])
        noisy = to_chw_tensor(noisy_arr)
        return {"noisy": noisy, "stem": stem}


def make_train_val_split(
    data_root: str, val_fraction: float = 0.1, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Build a reproducible train/validation split over the matched pairs
    under `data_root`, using a fixed seed so the split is identical
    across runs and across train.py / evaluate.py / documentation.
    This is drawn ONLY from the 3,200 provided training pairs -- the
    official released test set must never be used here.
    """
    scan = scan_paired_dataset(data_root)
    if scan.missing_gt or scan.missing_noisy:
        raise ValueError(
            f"Cannot build a split: dataset at {data_root} has unmatched files "
            f"({len(scan.missing_gt)} + {len(scan.missing_noisy)}). "
            f"Run validate_dataset.py first."
        )

    import random

    stems = list(scan.matched)
    rng = random.Random(seed)
    rng.shuffle(stems)

    n_val = max(1, int(round(len(stems) * val_fraction)))
    val_stems = sorted(stems[:n_val])
    train_stems = sorted(stems[n_val:])
    return train_stems, val_stems