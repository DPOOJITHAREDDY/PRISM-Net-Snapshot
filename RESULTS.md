@'
# PRISM-Net — Final Results (50-Epoch Run)

## Training Summary
- **Model**: PRISM-Net, 497,364 trainable parameters
- **Training**: Full 50-epoch run, executed in two stages:
  - **Stage 1**: epochs 0-16 (includes an initial NaN divergence — see
    "Training Stability" below — and the fix that resolved it)
  - **Stage 2**: epochs 17-49, resumed from the Stage-1 checkpoint
- **Best checkpoint**: epoch 36
- **Best validation PSNR during training**: 28.176338003195696 dB (SSIM 0.7611130526904009)
- **Final epoch (49) validation**: PSNR 28.130669982486125 dB, SSIM 0.7606092311770708
- **Skipped/non-finite batches, epochs 5-49 (preserved history)**: 0
- **Device**: NVIDIA GPU (CUDA), PyTorch 2.11.0+cu130

## Complete Training History
The full epoch 0-49 trajectory is consolidated in one canonical file:
`checkpoints/training_history_final_50epochs.json`.

- **Epochs 0-4**: validation PSNR/SSIM recovered from TensorBoard event
  logs (the only surviving source for these epochs; the JSON history
  files begin at epoch 5). Train-loss component breakdown
  (Charbonnier/MS-SSIM/Sobel/FFT) for epochs 0-4 was **not captured** by
  any preserved source and is recorded as `null` rather than estimated.
- **Epoch 4 note**: epoch 4 was originally trained twice. The first
  attempt (before the numerical-stability fixes existed) produced NaN.
  The canonical history uses the **later, successful** epoch-4 value
  (PSNR 26.99598503112793, SSIM 0.7120012044906616), recorded after the
  fixes were applied and training was resumed from the epoch-3 checkpoint.
  The NaN value is excluded from the canonical trajectory and preserved
  only in the archived failed-run checkpoint (see Archival Artifacts).
- **Epochs 5-16**: copied verbatim from the preserved Stage-1 history
  (`checkpoints/epoch17_backup/training_history_epoch17.json`).
- **Epochs 17-49**: copied verbatim from the preserved Stage-2 history
  (`checkpoints/training_history.json`).

Verified programmatically: exactly 50 records, epochs 0-49 with no
duplicates or gaps, best epoch = 36, best PSNR = 28.176338003195696,
zero skipped/non-finite batches across epochs 5-49.

## Validation Results (Held-Out Split, 320 Images, seed=42)
Evaluated against `checkpoints/best_model.pth` (epoch 36) on the fixed
seed-42 validation split, which the model never trained on.

| Metric | Value |
|---|---|
| **PSNR** | 28.176338003195696 dB |
| **SSIM** | 0.7611130526904009 |
| **LPIPS (AlexNet)** | 0.2872775057679974 |
| **Mean inference time** | 45.59 ms/image |
| **Min inference time** | 40.96 ms |
| **Max inference time** | 978.41 ms |

Source: `experiments/val_metrics_final.json`. This is the canonical
final validation result — it matches `best_model.pth`'s own recorded
`best_val_psnr` to full float precision.

### Per-image examples (8 sampled validation images)
| Stem | PSNR | SSIM |
|---|---|---|
| 001570 | 28.32 | 0.534 |
| 002106 | 30.99 | 0.634 |
| 000272 | 20.31 | 0.728 |
| 000569 | 25.16 | 0.731 |
| 000064 | 28.96 | 0.900 |
| 001048 | 26.31 | 0.812 |
| 001695 | 26.75 | 0.878 |
| 001387 | 18.88 | 0.785 |

## Official Test Set Results (400 Images, No Ground Truth)
Inference-only run on the official KLA test set. **No PSNR/SSIM/LPIPS is
reported here** because ground truth is not distributed for this split —
any such number would be fabricated.

| Metric | Value |
|---|---|
| **Images processed** | 400/400 |
| **Failed** | 0 |
| **Mean inference time** | 56.37 ms/image |
| **Min inference time** | 41.09 ms |
| **Max inference time** | 397.32 ms |

Source: `experiments/test_inference_summary_final.json`.

## Training Stability
An earlier training attempt (Run 1, part of Stage 1) diverged to NaN
starting at epoch 4 (curriculum degradation severity ramping combined
with fp16 autocast on numerically sensitive layers). Root-caused and
fixed by:
- Computing `LayerNorm2d` internally in float32 (autocast-safe)
- Running the model's forward pass outside `autocast` (backward still
  uses `GradScaler` for speed)
- Reducing max curriculum severities (speckle/gaussian 0.35->0.15, blur
  0.50->0.30) and clamping augmented input to [-0.5, 2.5]
- Per-batch non-finite-loss detection with automatic skip-and-continue,
  and an automatic abort if more than 50% of an epoch's batches are
  non-finite

After these fixes, training was resumed from the epoch-3 checkpoint and
completed epochs 4-49 (spanning Stage 1's remainder and all of Stage 2)
with **zero skipped batches and zero non-finite losses** in the
preserved history (epochs 5-49; epoch 4 itself is a single recovered
value with no skip-count recorded, per the note above).

## Canonical Final Artifacts
These are the definitive, final-model outputs. Anything not listed here
is archival (see below).

- **Best weights**: `checkpoints/best_model.pth` (epoch 36)
- **Last checkpoint**: `checkpoints/last.pth` (epoch 49)
- **Complete training history**: `checkpoints/training_history_final_50epochs.json`
- **Validation metrics**: `experiments/val_metrics_final.json`
- **Test inference summary**: `experiments/test_inference_summary_final.json`
- **Validation restored images**: `outputs/val_restored/` (320 `.npy`)
- **Test restored images**: `outputs/test_restored_final/` (400 `.npy`)
- **Comparison panels**: `visualizations_final/` (12 `.png`)
  - 8 validation panels: NoisyLR | PRISM-Net Output (PSNR/SSIM) | GT
  - 4 official test panels: NoisyLR | PRISM-Net Output (no GT, OOD)

## Archival Artifacts (historical, NOT final results)
Copied (not moved) into `archive/`, with a manifest at
`archive/ARCHIVE_README.md`. Originals remain in place. These are kept
for traceability, not as candidates for use:

- `checkpoints/last_nan_run.pth` — the **failed** Run 1 checkpoint
  (epoch 49, best_val_psnr 25.868330693065737). Diverged to NaN at
  epoch 4, before the stability fixes existed. **Not part of the final
  successful trajectory.**
- `checkpoints/best_model_epoch15_backup.pth`, `checkpoints/epoch17_backup/`
  — intermediate epoch-14/epoch-16 snapshots, superseded by the final
  epoch-36/epoch-49 checkpoints.
- `experiments/val_metrics_clean.json`, `experiments/test_inference_summary.json`
  — validation/test results computed against the superseded epoch-14
  checkpoint, not `best_model.pth`.
- `outputs/test_restored/`, `visualizations/` — restored images and
  comparison panels generated from the epoch-14 checkpoint.
- `experiments/tensorboard/` — raw event files spanning all training
  attempts (failed and successful); the correct values from this data
  are merged into `training_history_final_50epochs.json`.

## Reproducing
```powershell
python scripts\materialize_val_split.py --data_root <TRAIN_ROOT> --output_dir experiments\val_split
python evaluate.py --input_dir experiments\val_split\NoisyLR --gt_dir experiments\val_split\GT --output_dir outputs\val_restored --weights checkpoints\best_model.pth --metrics_json experiments\val_metrics_final.json
python evaluate.py --input_dir <TEST_ROOT> --output_dir outputs\test_restored_final --weights checkpoints\best_model.pth --metrics_json experiments\test_inference_summary_final.json
```

## Limitations
- Validation PSNR/SSIM vary substantially per image (18.9-31.0 dB PSNR
  observed across just 8 sampled images) — restoration quality is
  content-dependent, not uniform.
- No GAN or perceptual-adversarial term was used by design (fidelity
  over hallucination), which likely caps LPIPS improvement relative to
  adversarially-trained restoration models.
- The 400-image official test set has no ground truth, so its numeric
  quality cannot be independently verified beyond visual inspection.
- Train-loss component values (Charbonnier/MS-SSIM/Sobel/FFT) for
  epochs 0-4 were not preserved in any recoverable source and are
  recorded as unavailable in the canonical history rather than
  estimated.
'@ | Out-File -FilePath RESULTS.md -Encoding utf8