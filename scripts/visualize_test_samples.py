#!/usr/bin/env python
"""
visualize_test_samples.py

Qualitative-only (no GT, no metrics) visualization of official test
set restorations. --seed here only controls WHICH test images are
sampled for display -- unrelated to the train/val split seed, since
the official test set is never split.

Usage:
    python scripts\\visualize_test_samples.py ^
        --test_dir "C:\\...\\Test_NoisyLR" ^
        --weights "checkpoints\\best_model.pth" ^
        --output_dir "visualizations" ^
        --num_samples 4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import PrismNetTestDataset
from src.model import PRISMNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
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

    test_ds = PrismNetTestDataset(args.test_dir)
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(test_ds), size=min(args.num_samples, len(test_ds)), replace=False)

    for idx in indices:
        sample = test_ds[int(idx)]
        stem = sample["stem"]
        noisy = sample["noisy"].unsqueeze(0).to(args.device)

        with torch.no_grad():
            pred = model(noisy).squeeze(0).cpu()

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        axes[0].imshow(np.clip(noisy.squeeze().cpu().numpy(), 0, 1), cmap="gray")
        axes[0].set_title("Degraded Test Input (NoisyLR)")
        axes[1].imshow(np.clip(pred.squeeze().numpy(), 0, 1), cmap="gray")
        axes[1].set_title("PRISM-Net Output\n(no GT available for this sample)")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Official test sample: {stem}")
        fig.tight_layout()
        fig.savefig(out_dir / f"test_comparison_{stem}.png", dpi=150)
        plt.close(fig)
        print(f"Saved visualizations/test_comparison_{stem}.png")

    print(f"\n{len(indices)} test-set qualitative panel(s) written to {out_dir}")


if __name__ == "__main__":
    main()