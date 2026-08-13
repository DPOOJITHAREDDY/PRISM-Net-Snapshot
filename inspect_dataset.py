#!/usr/bin/env python
"""
inspect_dataset.py

Reusable dataset-inspection script for PRISM-Net. Reports (never
fails) on a paired training root and/or a test root:
  - number of NoisyLR / GT files, matched pairs, missing pairs
  - per-array shape / dtype
  - intensity ranges (min/max) for NoisyLR and GT
  - files ignored as macOS metadata (__MACOSX, ._*, .DS_Store)
  - any corrupted / unreadable files

Usage (train root, containing NoisyLR/ and GT/):
    python inspect_dataset.py --data_root "C:\\path\\to\\train\\train"

Usage (test root, NoisyLR only, no GT):
    python inspect_dataset.py --test_dir "C:\\path\\to\\Test_NoisyLR"

Both together:
    python inspect_dataset.py --data_root "..." --test_dir "..."

Optional:
    --sample N            only compute per-array stats on N randomly
                           sampled files per folder (default: every file)
    --seed S               random seed used for --sample (default: 42)
    --report_json PATH     also write the full report as JSON
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import (
    array_stats,
    load_npy_array,
    scan_paired_dataset,
    scan_test_dataset,
)


def _sample(items, n, seed):
    if n is None or n >= len(items):
        return items
    rng = random.Random(seed)
    return rng.sample(items, n)


def _summarize_folder(name, stems, files_by_stem, sample_n, seed):
    print(f"\n--- {name} ---")
    print(f"Files found: {len(stems)}")
    stems_to_check = _sample(list(stems), sample_n, seed)

    shapes, dtypes = {}, {}
    mins, maxs = [], []
    corrupted, nan_files, inf_files = [], [], []

    for stem in stems_to_check:
        path = files_by_stem[stem]
        try:
            arr = load_npy_array(path)
        except ValueError as exc:
            corrupted.append((stem, str(exc)))
            continue
        stats = array_stats(arr)
        shapes[stats["shape"]] = shapes.get(stats["shape"], 0) + 1
        dtypes[stats["dtype"]] = dtypes.get(stats["dtype"], 0) + 1
        mins.append(stats["min"])
        maxs.append(stats["max"])
        if stats["has_nan"]:
            nan_files.append(stem)
        if stats["has_inf"]:
            inf_files.append(stem)

    print(f"Inspected: {len(stems_to_check)}")
    print(f"Shape distribution: {shapes}")
    print(f"Dtype distribution: {dtypes}")
    if mins:
        print(f"Intensity range: min={min(mins):.5f}, max={max(maxs):.5f}")
    if corrupted:
        print(f"CORRUPTED / UNREADABLE FILES: {len(corrupted)}")
        for stem, err in corrupted[:10]:
            print(f"  - {stem}: {err}")
    if nan_files:
        print(f"Files containing NaN: {len(nan_files)} (e.g. {nan_files[:5]})")
    if inf_files:
        print(f"Files containing Inf: {len(inf_files)} (e.g. {inf_files[:5]})")

    return {
        "num_files": len(stems),
        "num_inspected": len(stems_to_check),
        "shape_distribution": {str(k): v for k, v in shapes.items()},
        "dtype_distribution": dtypes,
        "min": min(mins) if mins else None,
        "max": max(maxs) if maxs else None,
        "corrupted_files": corrupted,
        "nan_files": nan_files,
        "inf_files": inf_files,
    }


def inspect_train_root(data_root, sample_n, seed):
    print("=" * 60)
    print(f"TRAIN ROOT: {data_root}")
    print("=" * 60)

    result = scan_paired_dataset(data_root)

    print(f"NoisyLR files found: {len(result.noisy_files)}")
    print(f"GT files found:      {len(result.gt_files)}")
    print(f"Matched pairs:       {result.num_matched}")
    print(f"Missing GT (NoisyLR w/o GT):      {len(result.missing_gt)}")
    if result.missing_gt:
        print(f"  e.g. {result.missing_gt[:5]}")
    print(f"Missing NoisyLR (GT w/o NoisyLR): {len(result.missing_noisy)}")
    if result.missing_noisy:
        print(f"  e.g. {result.missing_noisy[:5]}")
    print(f"Ignored metadata files in NoisyLR/: {len(result.ignored_noisy)}")
    print(f"Ignored metadata files in GT/:      {len(result.ignored_gt)}")

    noisy_report = _summarize_folder(
        "NoisyLR", result.noisy_files.keys(), result.noisy_files, sample_n, seed
    )
    gt_report = _summarize_folder(
        "GT", result.gt_files.keys(), result.gt_files, sample_n, seed
    )

    return {
        "data_root": str(data_root),
        "num_noisy_files": len(result.noisy_files),
        "num_gt_files": len(result.gt_files),
        "num_matched_pairs": result.num_matched,
        "missing_gt": result.missing_gt,
        "missing_noisy": result.missing_noisy,
        "ignored_noisy_count": len(result.ignored_noisy),
        "ignored_gt_count": len(result.ignored_gt),
        "noisy": noisy_report,
        "gt": gt_report,
    }


def inspect_test_root(test_dir, sample_n, seed):
    print("=" * 60)
    print(f"TEST ROOT: {test_dir}")
    print("=" * 60)

    files, ignored = scan_test_dataset(test_dir)
    print(f"Test NoisyLR files found: {len(files)}")
    print(f"Ignored metadata files:   {len(ignored)}")

    report = _summarize_folder("Test NoisyLR", files.keys(), files, sample_n, seed)
    return {
        "test_dir": str(test_dir),
        "num_files": len(files),
        "ignored_count": len(ignored),
        "report": report,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect PRISM-Net dataset(s).")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--sample", type=int, default=None,
                         help="Only inspect N sampled files per folder (default: all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_json", type=str, default=None)
    args = parser.parse_args()

    if args.data_root is None and args.test_dir is None:
        parser.error("Provide at least one of --data_root or --test_dir")

    full_report = {}
    if args.data_root:
        full_report["train"] = inspect_train_root(args.data_root, args.sample, args.seed)
    if args.test_dir:
        full_report["test"] = inspect_test_root(args.test_dir, args.sample, args.seed)

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)

    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump(full_report, f, indent=2)
        print(f"Full JSON report written to: {args.report_json}")


if __name__ == "__main__":
    main()