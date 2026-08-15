# Submission Notes

This project implements a compact residual U-Net style image restoration model for semiconductor inspection data. It takes a noisy, downsampled 128x128 grayscale input and produces a restored 256x256 grayscale output in a single forward pass.

## Key strengths

- Handles grayscale inputs only
- Handles speckle noise and low-resolution degradation jointly
- Uses a compact architecture for speed-oriented inference
- Uses a combined loss with L1 + Charbonnier + gradient-preservation terms
- Saves checkpoints and generates validation artifacts
- Includes inference script for real image restoration

## Verified status

The project has been validated on the real attached dataset with successful runs for:

- dataset pairing check
- model forward pass
- training loop on real subsets
- checkpoint save
- validation metric generation
- inference on a real noisy sample

## Caveat

This environment has no CUDA device, so the verified runs were performed on CPU. The project is therefore ready as a working demonstration pipeline, but a final GPU benchmark and full long-run training campaign would require a CUDA-capable environment.
