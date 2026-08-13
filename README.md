# PRISM-Net -- Frozen 50-Epoch Snapshot

This repository is a FROZEN SNAPSHOT of the completed 50-epoch
PRISM-Net training run. It is not the active experiment repository
and will not be updated for future training runs (65/80/etc epochs).

## What this snapshot represents
- Full 50-epoch training run (epochs 0-49), executed in two stages
  (Stage 1: epochs 0-16, Stage 2: epochs 17-49; Stage 1 includes an
  initial NaN divergence that was root-caused and fixed -- see
  RESULTS.md for details)
- Best checkpoint: epoch 36
- Best validation PSNR: 28.176338003195696 dB
- Validation SSIM: 0.7611130526904009
- Validation LPIPS: 0.2872775057679974
- Validation set: 320 images (seed=42, held out)
- Official test set: 400/400 images restored successfully, no ground
  truth available (PSNR/SSIM/LPIPS not applicable)

## Canonical files in this snapshot
- `checkpoints/best_model.pth` -- the final trained model (epoch 36)
- `checkpoints/last.pth` -- end-of-run checkpoint (epoch 49)
- `checkpoints/training_history_final_50epochs.json` -- complete epoch
  0-49 training history
- `experiments/val_metrics_final.json` -- validation metrics
- `experiments/test_inference_summary_final.json` -- test inference summary
- `experiments/val_split/` -- the exact held-out validation split used
- `outputs/val_restored/`, `outputs/test_restored_final/` -- restored images

## What is intentionally excluded from this snapshot
- Intermediate/backup checkpoints (epoch 14, epoch 16, the failed NaN run)
- Audit and consolidation scripts/reports
- Presentation-generation scripts and the PPTX/presentation_assets/
- Raw TensorBoard event logs (already merged into the canonical history)
- The raw dataset

## Reproducing evaluation
```powershell
python scripts\materialize_val_split.py --data_root <TRAIN_ROOT> --output_dir experiments\val_split
python evaluate.py --input_dir experiments\val_split\NoisyLR --gt_dir experiments\val_split\GT --output_dir outputs\val_restored --weights checkpoints\best_model.pth --metrics_json experiments\val_metrics_final.json
```

See RESULTS.md for full results, architecture, and methodology.
