PRISM-Net -- Frozen 50-Epoch Project Snapshot

PRISM-Net is a degradation-aware image restoration and 2x super-resolution
network for degraded 128x128 grayscale images. The model restores each input
to a 256x256 output while explicitly adapting its restoration behavior to the
estimated degradation characteristics of the input.

This repository is the frozen record of the completed 50-epoch PRISM-Net
experiment and its reproducibility/evaluation artifacts.

Quick Overview

Problem

The task is to restore degraded low-resolution grayscale images affected by
noise and information loss while simultaneously producing a 2x higher
resolution output.

Input

NoisyLR image: 128 x 128 grayscale

Output

Restored image: 256 x 256 grayscale

Core PRISM-Net ideas

Degradation-aware conditioning

Dual-domain processing using raw intensity and a log-domain representation

FiLM-conditioned restoration blocks

NAF-style attention-free restoration blocks

PixelShuffle-based 2x super-resolution

Global bicubic residual connection

Numerically safe handling of raw and log-domain inputs

KLA Submission Package

A self-contained inference package is included for KLA evaluation:

KLA_Submission/
|-- README.md
|-- requirements.txt
|-- run.py
`-- models/
    `-- best_model.pth

The package contains everything required for standalone inference:

File

Purpose

KLA_Submission/run.py

Standalone PRISM-Net inference runner

KLA_Submission/requirements.txt

Pinned inference dependencies

KLA_Submission/README.md

Standalone KLA execution documentation

KLA_Submission/models/best_model.pth

Trained inference checkpoint

KLA Quick Start

From the KLA_Submission directory:

python run.py <input-dir> <output-dir>

Example:

python run.py input output

The runner:

Loads the included trained checkpoint.

Detects CUDA automatically when available.

Reads valid .npy grayscale inputs.

Runs PRISM-Net inference.

Produces 256x256 float32 restored outputs.

Preserves the input filename for the corresponding output.

Creates the output directory automatically when required.

The KLA package does not require an API key, external service, or model
weight download during inference.

KLA Verification

The standalone package was tested on the 320-image held-out validation
input set.

Verification result:

320/320 images processed successfully

0 failed images

320 output files generated

Output shape: 256x256

Output dtype: float32

All outputs finite

Global output range: [0, 1]

CUDA inference successfully verified

Total inference time for the 320-image validation run: 15.676 seconds
on the tested NVIDIA GeForce RTX 3050 6GB Laptop GPU

The standalone package also successfully processed individual test inputs
with the same output contract.

See KLA_Submission/README.md for the complete standalone instructions.

Frozen Experiment Results

This snapshot represents the completed 50-epoch training run, covering
epochs 0-49.

The run was executed in two stages:

Stage 1: epochs 0-16

Stage 2: epochs 17-49

Stage 1 included an initial NaN divergence that was root-caused and fixed.
The documented experimental history is available in RESULTS.md.

Best Validation Checkpoint

The best checkpoint was selected using validation PSNR.

Metric

Result

Best checkpoint epoch

36

Best validation PSNR

28.176338003195696 dB

Validation SSIM

0.7611130526904009

Validation LPIPS

0.2872775057679974

Validation set size

320 images

Validation split seed

42

The primary checkpoint is:

checkpoints/best_model.pth

The final end-of-run checkpoint is:

checkpoints/last.pth

Official Test Inference

The official test set contains 400 images and does not provide ground-truth
images.

The frozen PRISM-Net model successfully generated restored outputs for:

400 / 400 images

The restored outputs are stored in:

outputs/test_restored_final/

The corresponding inference summary is:

experiments/test_inference_summary_final.json

Because ground truth is unavailable for the official test set, PSNR, SSIM
and LPIPS are not applicable to those outputs.

Model Architecture

PRISM-Net is organized as an end-to-end restoration pipeline:

Input: 128x128 raw NoisyLR
          |
          v
Degradation Estimation Head
          |
          |----> noise-level descriptor
          |----> degradation-severity descriptor
          |
          v
Conditioning Vector
          |
          v
Dual-Domain Denoising Trunk
    |                 |
    |                 +--> log(1 + x) domain
    |
    +--------------------> raw domain with soft-range stabilization
          |
          v
FiLM-conditioned NAF-style restoration blocks
          |
          v
Super-Resolution Head
          |
          v
PixelShuffle 2x upsampling
          |
          v
FiLM-conditioned refinement blocks
          |
          v
Learned residual
          |
          +---- Bicubic 2x baseline
                    |
                    v
             Final 256x256 output

1. Degradation Estimation Head

The degradation estimation head analyzes the raw degraded input and predicts
two image-specific descriptors:

Estimated noise level

Estimated degradation severity

These descriptors are embedded into a conditioning vector.

The conditioning vector is then consumed by FiLM layers throughout the
denoising trunk and super-resolution head.

This allows the network to condition its restoration behavior on the
estimated degradation characteristics of each input rather than applying
one fixed restoration transform to every image.

2. Dual-Domain Denoising Trunk

The denoising trunk uses two complementary representations:

Raw-domain representation
Log-domain representation: log(1 + x)

The raw-domain branch retains the original measured intensity information.

A learnable soft-range clamp is used to stabilize overshoot values without
using hard clipping.

The log-domain branch provides a complementary representation for handling
multiplicative speckle-like degradation.

The raw and log representations are combined and processed by a stack of
FiLM-conditioned NAF-style restoration blocks.

3. NAF-Style Restoration Blocks

The restoration blocks are attention-free and use:

LayerNorm2d

1x1 convolutions

Depthwise 3x3 convolution

SimpleGate

Simplified channel attention

Residual connections

Learnable residual scaling

FiLM conditioning

The design focuses on efficient image restoration without conventional
self-attention.

4. Super-Resolution Head

The super-resolution head receives the denoised feature representation and
performs 2x spatial upsampling using PixelShuffle.

The upsampled features are then processed by additional FiLM-conditioned
NAF-style refinement blocks before producing the learned residual.

5. Global Residual Connection

PRISM-Net does not reconstruct the entire high-resolution image from
scratch.

Instead:

Final Output = Bicubic Upsampling + Learned Residual

The bicubic upsampled input provides a strong low-cost baseline, while the
network learns the restoration and detail correction required beyond that
baseline.

Final Model Configuration

The finalized inference checkpoint uses:

Parameter

Value

Input channels

1

Input resolution

128x128

Output channels

1

Output resolution

256x256

Feature width

48

Denoising blocks

8

Refinement blocks

4

Conditioning dimension

64

Super-resolution scale

2x

Numerical Safety and Preprocessing

The model is designed around the fact that degraded input values are not
guaranteed to remain inside [0, 1].

The observed NoisyLR range in the project materials was approximately:

-0.04995 to 1.68157

Therefore the preprocessing pipeline does not blindly clip the original
input to [0, 1].

For the log-domain representation, the transformation is:

log(1 + x)

A positive numerical floor is applied to the argument of the logarithm when
necessary so that out-of-distribution inputs cannot cause an invalid
logarithm.

The raw-domain branch remains based on the original input, while the
soft-range clamp is applied inside the denoising trunk for stabilization.

Repository Structure

PRISM-Net-Snapshot/
|
|-- checkpoints/
|   |-- best_model.pth
|   |-- last.pth
|   `-- training_history_final_50epochs.json
|
|-- configs/
|   `-- config.yaml
|
|-- experiments/
|   |-- val_split/
|   |-- val_metrics_final.json
|   `-- test_inference_summary_final.json
|
|-- outputs/
|   |-- test_restored_final/
|   `-- val_restored/
|
|-- scripts/
|   |-- materialize_val_split.py
|   |-- visualize_results.py
|   `-- visualize_test_samples.py
|
|-- src/
|   |-- augmentations.py
|   |-- blocks.py
|   |-- dataset.py
|   |-- degradation_head.py
|   |-- denoising_trunk.py
|   |-- losses.py
|   |-- metrics.py
|   |-- model.py
|   |-- preprocessing.py
|   |-- super_resolution.py
|   `-- utils.py
|
|-- tests/
|   `-- smoke_test.py
|
|-- KLA_Submission/
|   |-- README.md
|   |-- requirements.txt
|   |-- run.py
|   `-- models/
|       `-- best_model.pth
|
|-- evaluate.py
|-- train.py
|-- validate_dataset.py
|-- inspect_dataset.py
|-- README.md
|-- RESULTS.md
|-- requirements.txt
`-- LICENSE

Important Files

File / Folder

Purpose

checkpoints/best_model.pth

Best PRISM-Net checkpoint selected using validation PSNR

checkpoints/last.pth

Checkpoint from the end of the 50-epoch run

checkpoints/training_history_final_50epochs.json

Complete training and validation history

configs/config.yaml

Project configuration

experiments/val_split/

Frozen held-out validation split

experiments/val_metrics_final.json

Final validation metrics

experiments/test_inference_summary_final.json

Official test inference summary

outputs/val_restored/

Restored validation outputs

outputs/test_restored_final/

Restored official test outputs

src/model.py

Main PRISM-Net architecture

src/blocks.py

Shared restoration blocks

src/dataset.py

Dataset implementations

src/degradation_head.py

Degradation estimation head

src/denoising_trunk.py

Dual-domain denoising trunk

src/super_resolution.py

Super-resolution head

src/preprocessing.py

Numerical-safe preprocessing utilities

src/losses.py

Restoration loss implementation

src/metrics.py

PSNR, SSIM and LPIPS implementations

src/utils.py

General project utilities

evaluate.py

Standalone inference/evaluation interface

train.py

Training implementation

RESULTS.md

Detailed architecture, methodology and experimental analysis

requirements.txt

Frozen training environment dependencies

KLA_Submission/

Self-contained KLA inference package

Reproducing Evaluation

The official raw dataset is intentionally not included in this repository.

When the original dataset is available, the frozen validation split can be
materialized using:

python scripts\materialize_val_split.py --data_root <TRAIN_ROOT> --output_dir experiments\val_split

Validation evaluation can then be run with:

python evaluate.py --input_dir experiments\val_split\NoisyLR --gt_dir experiments\val_split\GT --output_dir outputs\val_restored --weights checkpoints\best_model.pth --metrics_json experiments\val_metrics_final.json

Standalone Inference

For inference without ground truth:

python evaluate.py --input_dir <INPUT_DIR> --output_dir <OUTPUT_DIR> --weights checkpoints\best_model.pth

The evaluation interface:

Loads the supplied checkpoint.

Reads the architecture parameters stored in the checkpoint when available.

Discovers valid .npy input files.

Runs PRISM-Net inference.

Produces 256x256 float32 outputs.

Preserves input filename stems.

Writes outputs to the requested directory.

Reports processed files and inference timing.

Optionally computes PSNR, SSIM and LPIPS when --gt_dir is supplied.

Returns a non-zero exit status if inference failures occur.

For metric evaluation:

python evaluate.py --input_dir <INPUT_DIR> --gt_dir <GT_DIR> --output_dir <OUTPUT_DIR> --weights checkpoints\best_model.pth --metrics_json <METRICS_JSON>

Run:

python evaluate.py --help

for the complete command-line interface.

Training

The main training implementation is:

train.py

The architecture and supporting components are located under:

src/

This repository records the completed 50-epoch experiment as a frozen
snapshot. It should be treated as a fixed record of the reported results.

Validation Protocol

The reported validation results were obtained using a held-out validation
set of 320 images.

The split uses:

Validation fraction: 10%

Split seed: 42

The best checkpoint is selected using validation PSNR.

Complete epoch-by-epoch history is stored in:

checkpoints/training_history_final_50epochs.json

Environment

The primary training environment included:

PyTorch 2.11.0 + CUDA 13.0

torchvision 0.26.0 + CUDA 13.0

NumPy 2.5.2

scikit-image 0.26.0

LPIPS 0.1.4

PyTorch-MSSSIM 1.0.0

TensorBoard 2.21.0

The exact frozen Python environment is recorded in:

requirements.txt

The KLA package has its own minimal inference dependency specification:

KLA_Submission/requirements.txt

with the tested versions:

numpy==2.5.2
torch==2.11.0+cu130

What Is Intentionally Excluded

The following temporary or non-canonical artifacts are intentionally not
part of the frozen project snapshot:

Intermediate and backup checkpoints

Failed NaN-run artifacts

Audit and consolidation scripts/reports

Presentation-generation scripts

PPTX and presentation assets

Raw TensorBoard event logs

Raw dataset

The repository focuses on the completed model, source implementation,
evaluation interface, canonical checkpoints, reported results, restored
outputs and reproducibility information.

Limitations

The official test set does not provide ground truth, so quantitative
PSNR, SSIM and LPIPS evaluation cannot be performed for those 400 images.

The repository does not include the raw dataset.

The KLA package is intended for the specified grayscale 128x128 input
format and produces 256x256 outputs.

Performance outside the training/data distribution should be evaluated
separately.

Project Status

This repository contains the completed frozen 50-epoch PRISM-Net experiment
and the associated evaluation artifacts.

The KLA-ready inference package has been independently verified on the
320-image held-out validation input set with zero inference failures.

The project also includes the final project presentation and supporting
experiment artifacts in the repository history.

For detailed experimental history, methodology, architecture decisions and
result analysis, see:

RESULTS.md

For standalone KLA execution instructions, see:

KLA_Submission/README.md

KLA Submission Package -- Detailed Reference

Package Contents

KLA_Submission/
|-- README.md
|-- requirements.txt
|-- run.py
`-- models/
    `-- best_model.pth

Input Contract

Each input file must be a NumPy .npy file containing a grayscale degraded
image with a 128x128 spatial resolution.

Supported array shapes:

(128, 128)
(1, 128, 128)

Output Contract

For each valid input, the runner creates an output file with the same
filename stem.

Output contract:

shape: (256, 256)
dtype: float32
range: [0, 1]

All verified outputs were finite and contained no NaN or Inf values.

Tested Inference Result

The complete held-out validation inference test produced:

Input files:          320
Processed:            320
Failed:               0
Output files:         320
Output shape:         256 x 256
Output dtype:         float32
Global minimum:       0.0
Global maximum:       1.0
Validation status:    PASS
Total inference time: 15.676 seconds

The tested hardware was:

NVIDIA GeForce RTX 3050 6GB Laptop GPU

The package is therefore ready as a standalone inference submission in
addition to the full project repository.

Final Artifact Map

For a reviewer or evaluator, the most important artifacts are:

Full project:
README.md

Detailed technical documentation:
RESULTS.md

Best trained model:
checkpoints/best_model.pth

Standalone evaluation:
evaluate.py

KLA-ready standalone package:
KLA_Submission/

KLA model:
KLA_Submission/models/best_model.pth

KLA runner:
KLA_Submission/run.py

KLA instructions:
KLA_Submission/README.md

KLA dependencies:
KLA_Submission/requirements.txt

This README documents the frozen PRISM-Net project, its reported 50-epoch
results, its evaluation interface, and its standalone KLA submission
package.