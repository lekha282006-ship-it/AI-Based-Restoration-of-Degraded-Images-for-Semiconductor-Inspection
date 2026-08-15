# AI-Based Restoration of Degraded Semiconductor Inspection Images

This project restores degraded semiconductor inspection images by learning a direct mapping from noisy 128x128 grayscale inputs to clean 256x256 target outputs. The model performs denoising, deblurring, and 2x super-resolution in a single forward pass.

## 1. Project objective

The goal is to restore degraded semiconductor wafer or inspection images so that hidden structural features become visually clearer and more usable for downstream inspection. The implemented model is designed for a compact, fast restoration pipeline suitable for CPU-only experimentation and validation.

## 2. Dataset and split

The real attached dataset is organized as follows:

- `train/train/NoisyLR/`: noisy low-resolution grayscale `.npy` files
- `train/train/GT/`: clean high-resolution grayscale `.npy` files
- `Test_NoisyLR/NoisyLR/`: held-out noisy inputs not used in the training split

The files are paired by filename stem, for example `000000.npy` in both the noisy and GT folders. The training pipeline uses a validation split from the training folder, while the external test folder is used to test generalization on previously unseen semiconductor structures.

## 3. Preprocessing and model

The degraded inputs may exceed the nominal `[0, 1]` range because of noise. The preprocessing therefore applies:

- clipping to `[0.0, 1.5]`
- scaling by `1.5`

This keeps the network numerically stable while preserving the artifact structure that must be removed.

The model is a compact residual CNN with:

- residual blocks
- instance normalization
- pixel-shuffle upscaling head
- final output clamp to `[0, 1]`

This is intentionally lightweight to allow reasonable CPU execution time while still performing restoration in a single pass.

## 4. Reproduction steps for a judge

### 4.1 Setup

From the project root:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4.2 Train the model

```bash
.venv\Scripts\python.exe train.py --epochs 5 --batch-size 8 --lr 2e-4
```

This produces a checkpoint in `checkpoints/demo_model.pth` and writes validation summaries under `results/`.

### 4.3 Run a single-image restoration

```bash
.venv\Scripts\python.exe inference.py --input-path "C:\path\to\noisy.npy" --output-path "results\restored.npy" --checkpoint "checkpoints\demo_model.pth"
```

### 4.4 Run folder-level evaluation on paired train data

```bash
.venv\Scripts\python.exe evaluate.py --input-dir "C:\Users\Welcome\Downloads\train\train\NoisyLR" --gt-dir "C:\Users\Welcome\Downloads\train\train\GT" --checkpoint "checkpoints\demo_model.pth" --output-dir "results\evaluation"
```

Optional quick validation on a small subset:

```bash
.venv\Scripts\python.exe evaluate.py --input-dir "C:\Users\Welcome\Downloads\train\train\NoisyLR" --gt-dir "C:\Users\Welcome\Downloads\train\train\GT" --checkpoint "checkpoints\demo_model.pth" --output-dir "results\evaluation_quick" --max-samples 2
```

### 4.5 Run unseen-structure generalization check

This is the key OOD/generalization proof in the project. The model is trained on the paired train split and then evaluated on the external `Test_NoisyLR/NoisyLR` folder without any retraining or fine-tuning.

```bash
.venv\Scripts\python.exe inference.py --input-dir "C:\Users\Welcome\Downloads\Test_NoisyLR\NoisyLR" --output-dir "results\test_ood_predictions" --checkpoint "checkpoints\demo_model.pth"
```

This checks the model on previously unseen semiconductor defect patterns and saves restored outputs for all files in the external folder. Because this folder is not used in the optimization phase and contains different degraded patterns from the training data, the run serves as an explicit generalization test.

> Note: the public test folder does not include paired GT labels, so the generalization proof is reported as a qualitative held-out inference check rather than a numeric OOD benchmark with PSNR/SSIM/LPIPS on unseen labels.

## 5. Judge-facing screenshots and visual gallery

The validation screenshot below is directly embedded in the project README so reviewers can see the restoration quality on the GitHub page itself.

![Validation comparison grid for semiconductor restoration](results/val_grid_demo.png)

The following outputs are useful for presentation slides and project review screens:

- `results/val_grid_demo.png`: side-by-side degraded / restored / target comparison for validation
- `results/test_ood_predictions/`: restored predictions on the held-out semiconductor test folder
- `results/evaluation_quick/`: quick folder-level evaluation snapshots with metrics summary

A typical slide-ready set for a reviewer is:

1. Input degraded semiconductor image
2. Model output after restoration
3. Ground-truth target comparison
4. Validation metric summary (PSNR / SSIM / LPIPS)
5. Generalization example on external unseen patterns

You can reuse the PNG files in this repository or export additional screenshots for your final PPT deck.

## 6. Output artifacts

- `checkpoints/demo_model.pth`: trained weights
- `results/validation_summary.json`: validation metrics from the demo training pipeline
- `results/val_grid_demo.png`: validation comparison grid
- `results/inference_sample.npy`: example restored output
- `results/evaluation/metrics.json`: PSNR/SSIM/LPIPS summary for paired validation folders
- `results/test_ood_predictions/`: restored predictions on the external held-out test set
- `submission_notes.md`: project summary for review

## 7. Verified status

The project has been validated on the real dataset with successful runs for:

- dataset pairing validation
- model forward pass
- training loop and checkpoint saving
- validation metric generation
- single-image inference
- folder-level evaluation with LPIPS, SSIM, and PSNR
- held-out folder inference on unseen test inputs

## 8. Environment note

This workspace is CPU-only, so the experiments were run on CPU rather than CUDA. The project is therefore validated as a compact, working restoration pipeline rather than as a GPU-optimized production benchmark.
