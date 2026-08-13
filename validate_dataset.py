#!/usr/bin/env python
"""
validate_dataset.py

Strict pre-training / pre-evaluation dataset validator for PRISM-Net.

Unlike inspect_dataset.py (reports, never fails), this script exits
non-zero the moment it finds a problem that would break training or
evaluation.

Checks on a training root (--data_root):
  - NoisyLR/ and GT/ subfolders exist
  - every NoisyLR file has a matching GT file and vice versa (0 missing
    required)
  - every checked array loads without error
  - NoisyLR arrays are 2D, dtype float32, shape (128, 128)
  - GT arrays are 2D, dtype float32, shape (256, 256)
  - no NaN / Inf values
  - GT values lie within [0, 1] (as documented by the challenge)

Checks on a test root (--test_dir):
  - contains valid NoisyLR files
  - every checked array loads without error, is 2D, float32, (128, 128)

Exit code 0 = all checks passed. Exit code 1 = at least one check failed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import (
    array_stats,
    load_npy_array,
    scan_paired_dataset,
    scan_test_dataset,
)

EXPECTED_NOISY_SHAPE = (128, 128)
EXPECTED_GT_SHAPE = (256, 256)
EXPECTED_DTYPE = "float32"
GT_VALID_RANGE = (0.0, 1.0)
GT_RANGE_TOLERANCE = 1e-4


class ValidationError(Exception):
    pass


def _check_array(path, expected_shape, check_gt_range: bool):
    arr = load_npy_array(path)
    stats = array_stats(arr)

    if arr.ndim != 2:
        raise ValidationError(f"{path}: expected a 2D array, got shape {arr.shape}")
    if stats["shape"] != expected_shape:
        raise ValidationError(f"{path}: expected shape {expected_shape}, got {stats['shape']}")
    if stats["dtype"] != EXPECTED_DTYPE:
        raise ValidationError(f"{path}: expected dtype {EXPECTED_DTYPE}, got {stats['dtype']}")
    if stats["has_nan"]:
        raise ValidationError(f"{path}: contains NaN values")
    if stats["has_inf"]:
        raise ValidationError(f"{path}: contains Inf values")
    if check_gt_range:
        lo, hi = GT_VALID_RANGE
        if stats["min"] < lo - GT_RANGE_TOLERANCE or stats["max"] > hi + GT_RANGE_TOLERANCE:
            raise ValidationError(
                f"{path}: GT value out of documented [0, 1] range "
                f"(min={stats['min']:.5f}, max={stats['max']:.5f})"
            )
    return stats


def validate_train_root(data_root, full: bool):
    print(f"Validating training root: {data_root}")
    result = scan_paired_dataset(data_root)

    if not result.noisy_dir.is_dir():
        raise ValidationError(f"Missing NoisyLR directory: {result.noisy_dir}")
    if not result.gt_dir.is_dir():
        raise ValidationError(f"Missing GT directory: {result.gt_dir}")
    if len(result.noisy_files) == 0:
        raise ValidationError(f"No .npy files found in {result.noisy_dir}")
    if len(result.gt_files) == 0:
        raise ValidationError(f"No .npy files found in {result.gt_dir}")
    if result.missing_gt:
        raise ValidationError(
            f"{len(result.missing_gt)} NoisyLR file(s) have no matching GT file, "
            f"e.g. {result.missing_gt[:5]}"
        )
    if result.missing_noisy:
        raise ValidationError(
            f"{len(result.missing_noisy)} GT file(s) have no matching NoisyLR file, "
            f"e.g. {result.missing_noisy[:5]}"
        )

    print(f"  {result.num_matched} matched pairs found. Checking arrays...")
    stems = result.matched if full else result.matched[: min(200, len(result.matched))]
    if not full:
        print(f"  (--quick mode: checking {len(stems)} of {result.num_matched} pairs; use --full for an exhaustive check)")

    for i, stem in enumerate(stems):
        _check_array(result.noisy_files[stem], EXPECTED_NOISY_SHAPE, check_gt_range=False)
        _check_array(result.gt_files[stem], EXPECTED_GT_SHAPE, check_gt_range=True)
        if (i + 1) % 500 == 0:
            print(f"  checked {i + 1}/{len(stems)} pairs...")

    print(f"  OK: {len(stems)} pairs validated.")
    print(f"  Ignored metadata files: {len(result.ignored_noisy)} in NoisyLR/, {len(result.ignored_gt)} in GT/")
    return result.num_matched


def validate_test_root(test_dir, full: bool):
    print(f"Validating test root: {test_dir}")
    files, ignored = scan_test_dataset(test_dir)

    if len(files) == 0:
        raise ValidationError(f"No .npy files found under {test_dir}")

    stems = list(files.keys())
    check_stems = stems if full else stems[: min(200, len(stems))]
    if not full:
        print(f"  (--quick mode: checking {len(check_stems)} of {len(stems)} files; use --full for an exhaustive check)")

    for i, stem in enumerate(check_stems):
        _check_array(files[stem], EXPECTED_NOISY_SHAPE, check_gt_range=False)
        if (i + 1) % 100 == 0:
            print(f"  checked {i + 1}/{len(check_stems)} files...")

    print(f"  OK: {len(check_stems)} test files validated ({len(files)} total found).")
    print(f"  Ignored metadata files: {len(ignored)}")
    return len(files)


def main():
    parser = argparse.ArgumentParser(description="Strictly validate a PRISM-Net dataset.")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--full", action="store_true",
                         help="Check every file instead of the first 200 per folder.")
    parser.add_argument("--expect_train_pairs", type=int, default=None)
    parser.add_argument("--expect_test_count", type=int, default=None)
    args = parser.parse_args()

    if args.data_root is None and args.test_dir is None:
        parser.error("Provide at least one of --data_root or --test_dir")

    try:
        if args.data_root:
            n_pairs = validate_train_root(args.data_root, args.full)
            if args.expect_train_pairs is not None and n_pairs != args.expect_train_pairs:
                raise ValidationError(f"Expected {args.expect_train_pairs} training pairs, found {n_pairs}")
        if args.test_dir:
            n_test = validate_test_root(args.test_dir, args.full)
            if args.expect_test_count is not None and n_test != args.expect_test_count:
                raise ValidationError(f"Expected {args.expect_test_count} test files, found {n_test}")
    except (ValidationError, FileNotFoundError) as exc:
        print(f"\nVALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nALL VALIDATION CHECKS PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()