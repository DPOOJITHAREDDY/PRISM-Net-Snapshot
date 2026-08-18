# PRISM-Net - KLA Submission

## Overview

PRISM-Net is a degradation-aware image restoration and 2x super-resolution network designed to restore degraded 128x128 grayscale images into 256x256 restored outputs.

The model combines:

- Degradation-aware conditioning
- Dual-domain processing using raw and log-transformed representations
- FiLM-conditioned NAF-style restoration blocks
- PixelShuffle-based 2x super-resolution
- A global bicubic residual connection

## Submission Structure

```text
KLA_Submission/
|-- run.py
|-- requirements.txt
|-- README.md
`-- models/
    `-- best_model.pth
```

## Requirements

- Python 3.x
- NumPy
- PyTorch

A CUDA-enabled NVIDIA GPU is used automatically when available.

No internet connection, API key, or external service is required during inference.

## Running the Model

From the `KLA_Submission` directory:

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py input output
```

The output directory is created automatically if it does not already exist.

## Input

The runner expects `.npy` files containing grayscale degraded images.

Expected input resolution:

```text
128 x 128
```

Supported input representations include:

- `(128, 128)`
- `(1, 128, 128)`

The runner processes all `.npy` files in the supplied input directory.

## Output

For every valid input file, PRISM-Net produces one `.npy` output using the same filename.

Output properties:

- Resolution: 256 x 256
- Format: `.npy`
- Data type: `float32`
- Single-channel grayscale
- Pixel values in the range `[0, 1]`
- No NaN or Inf values

Example:

```text
input/
|-- 000001.npy
|-- 000002.npy
`-- 000003.npy

output/
|-- 000001.npy
|-- 000002.npy
`-- 000003.npy
```

## Inference Pipeline

1. Load the trained model from `models/best_model.pth`.
2. Discover all `.npy` files in the input directory.
3. Automatically use CUDA when available.
4. Process each input image.
5. Restore the image from 128x128 to 256x256.
6. Validate the output shape and numerical values.
7. Save the restored output using the original filename.

## Model Configuration

The submitted checkpoint uses the finalized PRISM-Net configuration:

- Feature width: 48
- Denoising blocks: 8
- Refinement blocks: 4
- Conditioning dimension: 64
- Super-resolution scale: 2x

### Architecture

1. Degradation Estimation Head
2. Dual-Domain Denoising Trunk
3. Super-Resolution Head
4. Global Bicubic Residual Connection

## Model Design

### Degradation-Aware Conditioning

The Degradation Estimation Head analyzes each input and estimates two degradation descriptors:

- Noise level
- Degradation severity

These descriptors are embedded into a conditioning vector that is used by FiLM layers throughout the restoration network.

### Dual-Domain Processing

The denoising trunk processes two complementary representations:

- Raw-domain intensity
- Log-domain representation using `log(1 + x)`

The raw branch uses a learnable soft-range clamp for numerical stabilization, while the log branch is computed using a numerically safe transformation.

### Restoration Blocks

The network uses attention-free NAF-style restoration blocks containing:

- Layer normalization
- Depthwise convolution
- SimpleGate
- Simplified channel attention
- Residual connections
- FiLM conditioning

### Super-Resolution

The Super-Resolution Head uses PixelShuffle to perform 2x upsampling and applies additional FiLM-conditioned refinement blocks.

### Global Residual Connection

The model uses bicubic upsampling as a baseline and learns a residual correction on top of it:

```text
Final Output = Bicubic Upsampling + Learned Residual
```

This allows the network to focus primarily on restoration and detail reconstruction rather than learning the complete image from scratch.

## Reproducibility

The submission contains the complete inference implementation and trained checkpoint required to reproduce the restoration outputs from the supplied `.npy` inputs.

The model checkpoint is included locally at:

```text
models/best_model.pth
```

Inference does not require downloading model weights or accessing an external service.

## Inference Verification

The submission runner was tested on a held-out validation set of 320 images.

The verification produced:

- 320/320 images processed successfully
- 0 failed images
- 320 output files generated
- Output resolution: 256 x 256
- Output data type: `float32`
- All output values finite
- Output range: `[0, 1]`

The complete 320-image inference run completed successfully using CUDA.




