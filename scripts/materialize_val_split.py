#!/usr/bin/env python
"""
materialize_val_split.py

Copies the exact held-out validation .npy files (same --split_seed
split used by train.py) into their own NoisyLR/GT folder pair, so
evaluate.py can be pointed at ONLY unseen validation data for a clean
PSNR/SSIM/LPIPS number, rather than scoring against files the model
trained on.

IMPORTANT: --split_seed and --val_fraction here must match the values
train.py was run with (both default to seed=42, val_fraction=0.1;
only override both together, on both scripts, if you ever change them).

Usage:
    python scripts\\materialize_val_split.py --data_root "C:\\...\\train\\train" --output_dir "experiments\\val_split"
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import make_train_val_split
from src.utils import scan_paired_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    args = parser.parse_args()

    _, val_stems = make_train_val_split(args.data_root, val_fraction=args.val_fraction, seed=args.split_seed)
    scan = scan_paired_dataset(args.data_root)

    out_noisy = Path(args.output_dir) / "NoisyLR"
    out_gt = Path(args.output_dir) / "GT"
    out_noisy.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)

    for stem in val_stems:
        shutil.copy(scan.noisy_files[stem], out_noisy / f"{stem}.npy")
        shutil.copy(scan.gt_files[stem], out_gt / f"{stem}.npy")

    print(f"Materialized {len(val_stems)} validation pairs into {args.output_dir} "
          f"(split_seed={args.split_seed}, val_fraction={args.val_fraction})")


if __name__ == "__main__":
    main()