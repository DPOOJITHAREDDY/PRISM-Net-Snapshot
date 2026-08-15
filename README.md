# PRISM-Net -- Frozen 50-Epoch Snapshot

This repository is a FROZEN SNAPSHOT of the completed 50-epoch
PRISM-Net training run. It is not the active experiment repository
and will not be updated for future training runs (65/80/etc epochs).

## What this snapshot represents

- Full 50-epoch training run (epochs 0-49), executed in two stages
  (Stage 1: epochs 0-16, Stage 2: epochs 17-49; Stage 1 includes an
  initial NaN divergence that was root-caused and fixed -- see
  `RESULTS.md` for details)
- Best checkpoint: epoch 36
- Best validation PSNR: 28.176338003195696 dB
- Validation SSIM: 0.7611130526904009
- Validation LPIPS: 0.2872775057679974
- Validation set: 320 images (seed=42, held out)
- Official test set: 400/400 images restored successfully, no ground
  truth available (PSNR/SSIM/LPIPS not applicable)

---

## Submission Artifacts

The files required for evaluation can be located directly at the
following paths:

| Required Artifact | Location |
|---|---|
| **Trained model weights** | `checkpoints/best_model.pth` |
| **Standalone evaluation script** | `evaluate.py` |
| **Training script** | `train.py` |
| **Restored test outputs** | `outputs/test_restored_final/` |
| **Python environment** | `requirements.txt` |
| **Complete training history** | `checkpoints/training_history_final_50epochs.json` |
| **Detailed experimental results** | `RESULTS.md` |

### Model Selection

`checkpoints/best_model.pth` contains the model checkpoint selected
using validation PSNR.

- Best checkpoint epoch: **36**
- Best validation PSNR: **28.176338 dB**
- Validation SSIM: **0.761113**
- Validation LPIPS: **0.287278**

`checkpoints/last.pth` contains the checkpoint from the end of the
50-epoch training run.

---

## Repository Structure

```text
PRISM-Net-Snapshot/
│
├── checkpoints/
│   ├── best_model.pth
│   ├── last.pth
│   └── training_history_final_50epochs.json
│
├── configs/
│   └── config.yaml
│
├── experiments/
│   ├── val_split/
│   ├── val_metrics_final.json
│   └── test_inference_summary_final.json
│
├── outputs/
│   ├── test_restored_final/
│   └── val_restored/
│
├── scripts/
│   ├── materialize_val_split.py
│   ├── visualize_results.py
│   └── visualize_test_samples.py
│
├── src/
│   ├── augmentations.py
│   ├── blocks.py
│   ├── dataset.py
│   ├── degradation_head.py
│   ├── denoising_trunk.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── super_resolution.py
│   └── utils.py
│
├── tests/
│   └── smoke_test.py
│
├── evaluate.py
├── train.py
├── validate_dataset.py
├── inspect_dataset.py
├── README.md
├── RESULTS.md
├── requirements.txt
└── LICENSE
```

---

## File & Folder Purpose

| File / Folder | Purpose |
|---|---|
| `checkpoints/best_model.pth` | **Best PRISM-Net model checkpoint**, selected using validation PSNR. |
| `checkpoints/last.pth` | Checkpoint from the end of the 50-epoch training run. |
| `checkpoints/training_history_final_50epochs.json` | Complete training and validation history for epochs 0-49. |
| `configs/config.yaml` | Project configuration settings. |
| `experiments/val_split/` | Frozen held-out validation split used for evaluation. |
| `experiments/val_metrics_final.json` | Final validation evaluation metrics. |
| `experiments/test_inference_summary_final.json` | Summary of official test-set inference. |
| `outputs/test_restored_final/` | **400 restored outputs produced for the official test set.** |
| `outputs/val_restored/` | Restored outputs for the validation set. |
| `src/model.py` | Main PRISM-Net architecture definition. |
| `src/blocks.py` | Neural-network building blocks used by PRISM-Net. |
| `src/denoising_trunk.py` | Denoising and restoration trunk implementation. |
| `src/degradation_head.py` | Degradation estimation and conditioning component. |
| `src/super_resolution.py` | Super-resolution and reconstruction component. |
| `src/losses.py` | Composite restoration loss implementation. |
| `src/dataset.py` | Training, validation and test dataset implementations. |
| `src/preprocessing.py` | Image preprocessing and tensor conversion utilities. |
| `src/augmentations.py` | Training augmentation and degradation pipeline. |
| `src/metrics.py` | PSNR, SSIM and LPIPS metric implementations. |
| `src/utils.py` | General dataset and project utility functions. |
| `evaluate.py` | **Standalone inference/evaluation script used to generate restored outputs.** |
| `train.py` | Main PRISM-Net training script. |
| `validate_dataset.py` | Dataset validation and consistency checking. |
| `inspect_dataset.py` | Dataset inspection and analysis. |
| `scripts/materialize_val_split.py` | Creates the reproducible validation split. |
| `scripts/visualize_results.py` | Visualization of restoration results. |
| `scripts/visualize_test_samples.py` | Visualization of test samples/results. |
| `tests/smoke_test.py` | Basic project smoke test. |
| `RESULTS.md` | Detailed experimental results, architecture and methodology. |
| `requirements.txt` | **Exact frozen Python environment used for the training run.** |
| `LICENSE` | Project license. |

---

## Canonical Files in This Snapshot

- `checkpoints/best_model.pth` -- best trained model checkpoint
  selected using validation PSNR
- `checkpoints/last.pth` -- end-of-run checkpoint
- `checkpoints/training_history_final_50epochs.json` -- complete
  epoch 0-49 training history
- `experiments/val_metrics_final.json` -- final validation metrics
- `experiments/test_inference_summary_final.json` -- official test
  inference summary
- `experiments/val_split/` -- exact held-out validation split used
- `outputs/val_restored/` -- restored validation images
- `outputs/test_restored_final/` -- restored official test outputs

---

## Reproducing Evaluation

The official raw dataset is intentionally not included in this
repository.

For the held-out validation split, the following workflow can be used
when the original training dataset is available:

```powershell
python scripts\materialize_val_split.py --data_root <TRAIN_ROOT> --output_dir experiments\val_split

python evaluate.py --input_dir experiments\val_split\NoisyLR --gt_dir experiments\val_split\GT --output_dir outputs\val_restored --weights checkpoints\best_model.pth --metrics_json experiments\val_metrics_final.json
```

### Standalone Inference

`evaluate.py` can also be used independently for inference on a
directory containing NoisyLR `.npy` files.

```powershell
python evaluate.py --input_dir <INPUT_DIR> --output_dir <OUTPUT_DIR> --weights checkpoints\best_model.pth
```

The evaluation script:

1. Loads the supplied checkpoint.
2. Automatically reads the architecture parameters stored in the
   checkpoint.
3. Discovers valid `.npy` input files.
4. Runs PRISM-Net inference on all valid inputs.
5. Produces 256 x 256 float32 restored outputs.
6. Preserves the input filename stems.
7. Writes outputs to the specified output directory.
8. Reports processed files and inference timing.
9. Optionally computes PSNR, SSIM and LPIPS when a matching
   `--gt_dir` is supplied.
10. Returns a non-zero exit status if individual inference failures
    occur.

Optional metric evaluation:

```powershell
python evaluate.py --input_dir <INPUT_DIR> --gt_dir <GT_DIR> --output_dir <OUTPUT_DIR> --weights checkpoints\best_model.pth --metrics_json <METRICS_JSON>
```

See `evaluate.py --help` for all available command-line options.

---

## Training

The main training implementation is provided in:

```text
train.py
```

The model architecture and supporting components are located under:

```text
src/
```

The frozen snapshot represents the completed 50-epoch training run.
This repository is not intended to continue the later experimental
65/80+ epoch runs.

---

## Validation Results

The frozen model achieved:

| Metric | Result |
|---|---:|
| Best Validation PSNR | **28.176338 dB** |
| Validation SSIM | **0.761113** |
| Validation LPIPS | **0.287278** |
| Best Checkpoint Epoch | **36** |
| Validation Set Size | **320 images** |

The validation split uses:

- Validation fraction: **10%**
- Split seed: **42**

The complete epoch-by-epoch history is available in:

```text
checkpoints/training_history_final_50epochs.json
```

---

## Official Test Inference

The official test set contains **400 images** and does not provide
ground-truth images for evaluation.

The frozen PRISM-Net model successfully produced restored outputs
for:

**400 / 400 test images**

The restored outputs are stored in:

```text
outputs/test_restored_final/
```

The corresponding inference summary is stored in:

```text
experiments/test_inference_summary_final.json
```

Because ground truth is unavailable for the official test set,
PSNR, SSIM and LPIPS are not reported for those images.

---

## Environment

`requirements.txt` contains the exact `pip freeze` environment used
for the training run.

The primary training environment included:

- PyTorch 2.11.0 + CUDA 13.0
- torchvision 0.26.0 + CUDA 13.0
- NumPy 2.5.2
- scikit-image 0.26.0
- LPIPS 0.1.4
- PyTorch-MSSSIM 1.0.0
- TensorBoard 2.21.0

All package versions are pinned in `requirements.txt`.

---

## What Is Intentionally Excluded From This Snapshot

The following are intentionally excluded from the frozen repository:

- Intermediate/backup checkpoints
  (epoch 14, epoch 16 and the failed NaN run)
- Audit and consolidation scripts/reports
- Presentation-generation scripts
- PPTX and presentation assets
- Raw TensorBoard event logs
  (already consolidated into the canonical training history)
- Raw dataset

The snapshot is intended to provide the completed model, source
implementation, evaluation interface, results, restored outputs and
reproducibility information without including the raw dataset or
temporary experiment artifacts.

---

## Project Notes

This repository represents the **frozen 50-epoch PRISM-Net submission
artifact**.

It should be treated as a fixed record of the reported experiment.
Later training experiments are maintained separately and do not modify
the results represented here.

See `RESULTS.md` for the detailed architecture, methodology,
experimental history and result analysis.