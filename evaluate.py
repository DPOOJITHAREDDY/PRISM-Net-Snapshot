#!/usr/bin/env python
"""
evaluate.py

Standalone PRISM-Net evaluation / inference script. Used both for:
  (a) running inference on the official 400-image test set (no GT
      available -- only outputs + timing are produced), and
  (b) computing PSNR/SSIM/LPIPS against ground truth when a --gt_dir
      IS available (e.g. re-scoring the held-out validation split).

Required usage (per submission spec, must run with no source edits):
    python evaluate.py ^
        --input_dir "PATH_TO_INPUT" ^
        --output_dir "PATH_TO_OUTPUT" ^
        --weights "PATH_TO_WEIGHTS"

Optional (enables metric computation):
    --gt_dir "PATH_TO_GT"

Behavior:
  1. loads weights
  2. loads valid .npy inputs (ignores macOS metadata automatically)
  3. runs inference
  4. saves restored outputs as .npy, preserving input filenames
  5. produces 256x256 float32 outputs
  6. reports processed file count and per-image inference time
  7. fails clearly (non-zero exit, readable message) on invalid files
  8. requires no source-code editing -- everything is a CLI flag
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from src.dataset import PrismNetTestDataset
from src.metrics import compute_lpips, compute_psnr, compute_ssim
from src.model import PRISMNet
from src.preprocessing import to_chw_tensor
from src.utils import load_npy_array, scan_npy_directory


def parse_args():
    p = argparse.ArgumentParser(description="Run PRISM-Net inference / evaluation.")
    p.add_argument("--input_dir", type=str, required=True,
                    help="Directory of NoisyLR .npy files (or its parent containing NoisyLR/).")
    p.add_argument("--output_dir", type=str, required=True,
                    help="Directory to write restored .npy outputs into.")
    p.add_argument("--weights", type=str, required=True, help="Path to a .pth checkpoint.")
    p.add_argument("--gt_dir", type=str, default=None,
                    help="Optional: directory of matching GT .npy files (flat folder, matched by "
                         "filename stem), to compute PSNR/SSIM/LPIPS.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metrics_json", type=str, default=None,
                    help="Optional path to also write a JSON metrics/summary report.")
    # Fallback architecture hyperparameters -- only used if the checkpoint
    # has no saved "args" (e.g. an older/foreign checkpoint). Checkpoints
    # produced by this project's train.py always carry their own args and
    # override these automatically.
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--num_denoise_blocks", type=int, default=8)
    p.add_argument("--num_refine_blocks", type=int, default=4)
    p.add_argument("--cond_dim", type=int, default=64)
    return p.parse_args()


def load_model(weights_path, device, fallback_args):
    if not Path(weights_path).is_file():
        print(f"ERROR: weights file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    # weights_only=False: this checkpoint intentionally stores a plain
    # "args" dict alongside the tensors (see train.py::save_checkpoint).
    # torch>=2.6 defaults weights_only=True, which can reject that.
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    def get(key, default):
        return saved_args.get(key, default)

    model = PRISMNet(
        width=get("width", fallback_args.width),
        num_denoise_blocks=get("num_denoise_blocks", fallback_args.num_denoise_blocks),
        num_refine_blocks=get("num_refine_blocks", fallback_args.num_refine_blocks),
        cond_dim=get("cond_dim", fallback_args.cond_dim),
    ).to(device)

    if "model_state" not in ckpt:
        print(f"ERROR: checkpoint at {weights_path} has no 'model_state' key -- "
              f"is this a valid PRISM-Net checkpoint?", file=sys.stderr)
        sys.exit(1)

    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as exc:
        print(f"ERROR: checkpoint architecture does not match the model definition: {exc}",
              file=sys.stderr)
        sys.exit(1)

    model.eval()
    return model


def main():
    args = parse_args()
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = PrismNetTestDataset(args.input_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: could not load input directory: {exc}", file=sys.stderr)
        sys.exit(1)

    # GT discovery: --gt_dir is always treated as a flat folder of .npy
    # files, matched to predictions purely by filename stem -- the same
    # convention used everywhere else in this project (src/utils.py). No
    # assumption is made about --input_dir and --gt_dir sharing a parent
    # folder, so this works whether the two live side by side (as
    # materialize_val_split.py produces them) or anywhere else.
    gt_files = None
    if args.gt_dir:
        gt_files, ignored_gt = scan_npy_directory(args.gt_dir)
        print(f"Found {len(gt_files)} GT file(s) in {args.gt_dir} for scoring "
              f"({len(ignored_gt)} ignored metadata file(s))")

    model = load_model(args.weights, device, args)
    print(f"Loaded model with {model.count_parameters():,} parameters onto device: {device}")
    print(f"Found {len(dataset)} valid input file(s) in {args.input_dir}")

    per_image_times = []
    processed = 0
    failed = []
    metrics_accum = {"psnr": [], "ssim": [], "lpips": []}

    with torch.no_grad():
        for i in range(len(dataset)):
            stem = dataset.stems[i]
            try:
                sample = dataset[i]
                noisy = sample["noisy"].unsqueeze(0).to(device)  # (1,1,128,128)

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred = model(noisy)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                per_image_times.append(elapsed)

                pred_np = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
                if pred_np.shape != (256, 256):
                    raise ValueError(f"unexpected output shape {pred_np.shape}, expected (256, 256)")

                np.save(output_dir / f"{stem}.npy", pred_np)
                processed += 1

                if gt_files is not None and stem in gt_files:
                    gt_arr = load_npy_array(gt_files[stem])
                    # (1, 256, 256), NOT a bare (256, 256) -- compute_lpips
                    # relies on a channel dimension being present to
                    # decide whether to repeat grayscale to 3 channels.
                    gt_tensor = to_chw_tensor(gt_arr)
                    pred_single = pred.squeeze(0).cpu()  # (1, 256, 256)

                    metrics_accum["psnr"].append(compute_psnr(pred_single, gt_tensor))
                    metrics_accum["ssim"].append(compute_ssim(pred_single, gt_tensor))
                    metrics_accum["lpips"].append(
                        compute_lpips(pred_single, gt_tensor, device="cpu")
                    )

            except Exception as exc:  # noqa: BLE001 -- must fail clearly per-file, not crash the run
                failed.append((stem, str(exc)))
                print(f"  FAILED on '{stem}': {exc}", file=sys.stderr)

            if (i + 1) % 50 == 0:
                print(f"  processed {i + 1}/{len(dataset)}...")

    print(f"\nProcessed: {processed}/{len(dataset)}")
    if failed:
        print(f"Failed: {len(failed)} file(s):")
        for stem, err in failed[:20]:
            print(f"  - {stem}: {err}")

    summary = {"num_input_files": len(dataset), "num_processed": processed, "num_failed": len(failed)}

    if per_image_times:
        mean_time = sum(per_image_times) / len(per_image_times)
        summary["inference_time_sec"] = {
            "mean_per_image": mean_time,
            "min": min(per_image_times),
            "max": max(per_image_times),
            "total": sum(per_image_times),
        }
        print(f"Inference time per image: mean={mean_time*1000:.2f} ms, "
              f"min={min(per_image_times)*1000:.2f} ms, max={max(per_image_times)*1000:.2f} ms")

    if metrics_accum["psnr"]:
        summary["metrics"] = {
            "psnr_mean": sum(metrics_accum["psnr"]) / len(metrics_accum["psnr"]),
            "ssim_mean": sum(metrics_accum["ssim"]) / len(metrics_accum["ssim"]),
            "lpips_mean": sum(metrics_accum["lpips"]) / len(metrics_accum["lpips"]),
            "num_scored": len(metrics_accum["psnr"]),
        }
        print(f"Metrics over {summary['metrics']['num_scored']} scored image(s): "
              f"PSNR={summary['metrics']['psnr_mean']:.3f}  "
              f"SSIM={summary['metrics']['ssim_mean']:.4f}  "
              f"LPIPS={summary['metrics']['lpips_mean']:.4f}")
    else:
        print("No --gt_dir given (or no stem matches found) -- metrics not computed. "
              "This is expected/correct for the official test set, which has no GT.")

    if args.metrics_json:
        with open(args.metrics_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to: {args.metrics_json}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()