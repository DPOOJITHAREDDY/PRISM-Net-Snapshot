@'
# PRISM-Net -- Frozen 50-Epoch Snapshot

This repository is a FROZEN SNAPSHOT of the completed 50-epoch
PRISM-Net training run. It is not the active experiment repository
and will not be updated for future training runs.

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
| **KLA submission runner** | `KLA_Submission/run.py` |
| **KLA submission checkpoint** | `KLA_Submission/models/best_model.pth` |
| **KLA dependencies** | `KLA_Submission/requirements.txt` |
| **KLA documentation** | `KLA_Submission/README.md` |

### Model Selection

`checkpoints/best_model.pth` contains the model checkpoint selected
using validation PSNR.

- Best checkpoint epoch: **36**
- Best validation PSNR: **28.176338 dB**
- Validation SSIM: **0.761113**
- Validation LPIPS: **0.287278**

`checkpoints/last.pth` contains the checkpoint from the end of the
50-epoch training run.

The same checkpoint weights are also bundled standalone at
`KLA_Submission/models/best_model.pth` for evaluator convenience --
see the **KLA Submission Package** section below.

---

## KLA Submission Package

A self-contained inference package is provided under:

```text
KLA_Submission/
├── README.md
├── requirements.txt
├── run.py
└── models/
    └── best_model.pth
```

This package is a standalone inference/evaluation bundle intended to
let an evaluator run PRISM-Net without needing the rest of the
training repository or workflow -- no other files in this repository
are required to use it.

### `KLA_Submission/run.py`

Standalone inference runner. It:

1. Accepts an input directory
2. Accepts an output directory
3. Loads the bundled checkpoint
4. Automatically uses CUDA when available
5. Processes `.npy` grayscale inputs
6. Generates restored 256x256 outputs
7. Saves outputs using the original filename

### `KLA_Submission/models/best_model.pth`

The trained checkpoint bundled specifically with the KLA submission
(identical weights to `checkpoints/best_model.pth`, epoch 36).

### `KLA_Submission/requirements.txt`

The tested KLA inference dependencies:

```text
numpy==2.5.2
torch==2.11.0+cu130
```

### `KLA_Submission/README.md`

Standalone documentation for the KLA package, covering execution
instructions, input/output specification, and verification results
in full detail. The root README below summarizes the same package;
refer to `KLA_Submission/README.md` for the authoritative,
self-contained instructions an evaluator needs.

### Running the KLA package

From inside `KLA_Submission/`:

```powershell
python run.py <input-dir> <output-dir>
```

Example:

```powershell
python run.py input output
```

### KLA input specification

- Format: `.npy` grayscale degraded images
- Expected resolution: **128 x 128**
- Supported array shapes: `(128, 128)` or `(1, 128, 128)`
- All valid `.npy` files in the supplied input directory are processed

### KLA output specification

Each input produces one `.npy` output using the same filename.

- Resolution: **256 x 256**
- dtype: `float32`
- Single-channel grayscale
- Range: `[0, 1]`
- Finite values only -- no NaN, no Inf

### Verified KLA test

The KLA package was actually tested end-to-end against the held-out
validation input directory (320 images):

```text
320/320 images processed successfully
Failed: 0
320 output files generated
Total inference time: 15.676 seconds
```

Output verification across all 320 generated files:

```text
Files checked: 320
Shapes: {(256, 256)}
Dtypes: {float32}
Global min: 0.0
Global max: 1.0
Invalid outputs: 0
STATUS: PASS
```

### Hardware used for KLA verification

```text
Python 3.14.3
PyTorch: 2.11.0+cu130
CUDA: True
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

CUDA is used automatically when available; the package falls back to
CPU otherwise. The KLA package does not require an internet
connection, API key, or external service during inference, as
documented in `KLA_Submission/README.md`.

### KLA model configuration

```text
Feature width: 48
Denoising blocks: 8
Refinement blocks: 4
Conditioning dimension: 64
Super-resolution scale: 2x
```

Architecture components:

1. Degradation Estimation Head
2. Dual-Domain Denoising Trunk
3. Super-Resolution Head
4. Global Bicubic Residual Connection

### KLA model design

**Degradation-aware conditioning** -- The Degradation Estimation Head
estimates noise level and degradation severity, which are embedded
into a conditioning vector used by FiLM layers throughout the network.

**Dual-domain processing** -- The model processes both a raw-domain
intensity representation and a log-domain representation using
`log(1 + x)`. The raw branch uses a learnable soft-range clamp for
numerical stabilization; the log branch uses a numerically safe
transformation.

**Restoration blocks** -- NAF-style restoration blocks contain layer
normalization, depthwise convolution, SimpleGate, simplified channel
attention, residual connections, and FiLM conditioning.

**Super-resolution** -- PixelShuffle performs the 2x upsampling,
followed by additional FiLM-conditioned refinement blocks.

**Global residual** -- The final prediction follows:

```text
Final Output = Bicubic Upsampling + Learned Residual
```

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
├── KLA_Submission/
│   ├── README.md
│   ├── requirements.txt
│   ├── run.py
│   └── models/
│       └── best_model.pth
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
| `KLA_Submission/run.py` | Standalone KLA inference runner -- see **KLA Submission Package** above. |
| `KLA_Submission/models/best_model.pth` | Bundled checkpoint for the KLA package (epoch 36). |
| `KLA_Submission/requirements.txt` | Tested KLA inference dependencies (numpy, torch). |
| `KLA_Submission/README.md` | Standalone evaluator instructions for the KLA package. |
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
- `KLA_Submission/models/best_model.pth` -- standalone-bundled copy
  of the same checkpoint, verified against the full validation split

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

For an evaluator who only needs to run inference without the full
repository, see the **KLA Submission Package** above instead -- it
provides the same underlying model through a minimal, self-contained
interface (`KLA_Submission/run.py`).

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

The KLA Submission Package was independently verified against the
held-out validation split (320/320 images, 0 failures) -- see the
**KLA Submission Package** section above for the full verification
output.

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

The standalone `KLA_Submission/` package uses a minimal dependency
set (`numpy==2.5.2`, `torch==2.11.0+cu130`) -- see
`KLA_Submission/requirements.txt`.

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

Two README files exist in this repository, for different audiences:

- **`README.md`** (this file) -- comprehensive project documentation:
  architecture, training, results, evaluation, repository layout, and
  the KLA package overview.
- **`KLA_Submission/README.md`** -- standalone evaluator instructions
  for the KLA package specifically: how to run `run.py`, input/output
  specification, dependencies, checkpoint, and verification results.

Later training experiments are maintained separately and do not modify
the results represented here.

See `RESULTS.md` for the detailed architecture, methodology,
experimental history and result analysis.