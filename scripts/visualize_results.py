#!/usr/bin/env python
"""
visualize_results.py

Generates side-by-side (NoisyLR | Restored | GT) comparison panels for
qualitative evaluation, on the held-out validation split (same
--split_seed / --val_fraction convention as train.py and
materialize_val_split.py).

Usage:
    python scripts\\visualize_results.py ^
        --data_root "C:\\...\\train\\train" ^
        --weights "checkpoints\\best_model.pth" ^
        --output_dir "visualizations" ^
        --num_samples 8
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import PrismNetTrainDataset, make_train_val_split
from src.metrics import compute_psnr, compute_ssim
from src.model import PRISMNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--split_seed", type=int, default=42,
                         help="Must match the seed train.py's --split_seed used.")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                         help="Must match the value train.py's --val_fraction used.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # weights_only=False: this checkpoint stores a plain "args" dict
    # alongside the tensors -- torch>=2.6 defaults weights_only=True,
    # which can reject that.
    ckpt = torch.load(args.weights, map_location=args.device, weights_only=False)
    saved_args = ckpt.get("args", {})
    model = PRISMNet(
        width=saved_args.get("width", 48),
        num_denoise_blocks=saved_args.get("num_denoise_blocks", 8),
        num_refine_blocks=saved_args.get("num_refine_blocks", 4),
        cond_dim=saved_args.get("cond_dim", 64),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, val_stems = make_train_val_split(args.data_root, val_fraction=args.val_fraction, seed=args.split_seed)
    val_ds = PrismNetTrainDataset(args.data_root, stems=val_stems)

    rng = np.random.RandomState(args.split_seed)
    indices = rng.choice(len(val_ds), size=min(args.num_samples, len(val_ds)), replace=False)

    for idx in indices:
        sample = val_ds[int(idx)]
        stem = sample["stem"]
        noisy = sample["noisy"].unsqueeze(0).to(args.device)
        gt = sample["gt"]

        with torch.no_grad():
            pred = model(noisy).squeeze(0).cpu()

        psnr = compute_psnr(pred, gt)
        ssim = compute_ssim(pred, gt)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        axes[0].imshow(np.clip(noisy.squeeze().cpu().numpy(), 0, 1), cmap="gray")
        axes[0].set_title("Degraded Input (NoisyLR)")
        axes[1].imshow(np.clip(pred.squeeze().numpy(), 0, 1), cmap="gray")
        axes[1].set_title(f"PRISM-Net Output\nPSNR={psnr:.2f}  SSIM={ssim:.3f}")
        axes[2].imshow(gt.squeeze().numpy(), cmap="gray")
        axes[2].set_title("Ground Truth")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Sample: {stem}")
        fig.tight_layout()
        fig.savefig(out_dir / f"comparison_{stem}.png", dpi=150)
        plt.close(fig)
        print(f"Saved visualizations/comparison_{stem}.png  (PSNR={psnr:.2f}, SSIM={ssim:.3f})")

    print(f"\n{len(indices)} comparison panel(s) written to {out_dir}")


if __name__ == "__main__":
    main()