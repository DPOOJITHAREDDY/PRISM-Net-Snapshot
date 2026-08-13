"""
Shared dataset-scanning utilities for PRISM-Net.

Used by inspect_dataset.py, validate_dataset.py, and src/dataset.py so
the exact same file-discovery and pairing logic is used everywhere.
Metadata files produced by macOS zip extraction (__MACOSX/, ._*,
.DS_Store) are always ignored, never treated as data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

VALID_EXT = ".npy"


def _is_metadata(path: Path) -> bool:
    """Return True if `path` is macOS extraction metadata, not real data."""
    if path.name == ".DS_Store":
        return True
    if path.name.startswith("._"):
        return True
    if "__MACOSX" in path.parts:
        return True
    return False


def scan_npy_directory(directory: str) -> Tuple[Dict[str, Path], List[Path]]:
    """
    Scan `directory` for valid .npy files.

    Returns:
        files: dict mapping filename stem -> full Path, for every
            valid (non-metadata) .npy file found directly inside
            `directory`.
        ignored: list of Paths skipped because they looked like macOS
            metadata or had the wrong extension.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    files: Dict[str, Path] = {}
    ignored: List[Path] = []

    for entry in sorted(directory.iterdir()):
        if entry.is_dir():
            continue
        if _is_metadata(entry):
            ignored.append(entry)
            continue
        if entry.suffix.lower() != VALID_EXT:
            ignored.append(entry)
            continue
        stem = entry.stem
        if stem in files:
            raise ValueError(
                f"Duplicate stem '{stem}' found in {directory}: "
                f"{files[stem]} and {entry}"
            )
        files[stem] = entry

    return files, ignored


@dataclass
class PairedScanResult:
    noisy_dir: Path
    gt_dir: Path
    noisy_files: Dict[str, Path] = field(default_factory=dict)
    gt_files: Dict[str, Path] = field(default_factory=dict)
    matched: List[str] = field(default_factory=list)
    missing_gt: List[str] = field(default_factory=list)
    missing_noisy: List[str] = field(default_factory=list)
    ignored_noisy: List[Path] = field(default_factory=list)
    ignored_gt: List[Path] = field(default_factory=list)

    @property
    def num_matched(self) -> int:
        return len(self.matched)


def scan_paired_dataset(
    data_root: str, noisy_subdir: str = "NoisyLR", gt_subdir: str = "GT"
) -> PairedScanResult:
    """
    Scan a training-style dataset root containing `noisy_subdir/` and
    `gt_subdir/` subfolders, and match samples by filename stem.
    """
    data_root = Path(data_root)
    noisy_dir = data_root / noisy_subdir
    gt_dir = data_root / gt_subdir

    noisy_files, ignored_noisy = scan_npy_directory(str(noisy_dir))
    gt_files, ignored_gt = scan_npy_directory(str(gt_dir))

    noisy_stems = set(noisy_files.keys())
    gt_stems = set(gt_files.keys())

    matched = sorted(noisy_stems & gt_stems)
    missing_gt = sorted(noisy_stems - gt_stems)      # have NoisyLR, no GT
    missing_noisy = sorted(gt_stems - noisy_stems)   # have GT, no NoisyLR

    return PairedScanResult(
        noisy_dir=noisy_dir,
        gt_dir=gt_dir,
        noisy_files=noisy_files,
        gt_files=gt_files,
        matched=matched,
        missing_gt=missing_gt,
        missing_noisy=missing_noisy,
        ignored_noisy=ignored_noisy,
        ignored_gt=ignored_gt,
    )


def scan_test_dataset(
    test_root: str, noisy_subdir: str = "NoisyLR"
) -> Tuple[Dict[str, Path], List[Path]]:
    """Scan an inference-only test directory (NoisyLR only, no GT)."""
    test_root = Path(test_root)
    noisy_dir = test_root / noisy_subdir
    if noisy_dir.is_dir():
        return scan_npy_directory(str(noisy_dir))
    # Some callers point --input_dir directly at the NoisyLR folder itself.
    return scan_npy_directory(str(test_root))


def load_npy_array(path) -> np.ndarray:
    """Load a single .npy file, raising a clear error on any failure."""
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - want one unified clear error
        raise ValueError(f"Failed to load '{path}': {exc}") from exc
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"File '{path}' did not contain a numpy array.")
    return arr


def array_stats(arr: np.ndarray) -> dict:
    """Compute a small set of diagnostic statistics for one array."""
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())
    finite_vals = arr[np.isfinite(arr)]
    return {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(finite_vals.min()) if finite_vals.size else float("nan"),
        "max": float(finite_vals.max()) if finite_vals.size else float("nan"),
        "mean": float(finite_vals.mean()) if finite_vals.size else float("nan"),
        "has_nan": has_nan,
        "has_inf": has_inf,
    }