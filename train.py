import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import SemiconductorPairDataset, inspect_dataset, resolve_dataset_dirs
from src.model import LightSRNet
from src.utils import CombinedLoss, benchmark_inference, psnr, save_json, save_validation_grid, ssim


def parse_args():
    parser = argparse.ArgumentParser(description='Train a single-pass denoise + super-resolution model for semiconductor inspection images.')
    default_gt, default_lr = resolve_dataset_dirs()
    parser.add_argument('--train-gt', type=str, default=str(default_gt))
    parser.add_argument('--train-lr', type=str, default=str(default_lr))
    parser.add_argument('--val-gt', type=str, default=str(default_gt))
    parser.add_argument('--val-lr', type=str, default=str(default_lr))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--results-dir', type=str, default='results')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    inspect_dataset(args.train_gt, args.train_lr)

    train_dataset = SemiconductorPairDataset(args.train_gt, args.train_lr, split='train', val_fraction=0.1, augment=True)
    val_dataset = SemiconductorPairDataset(args.train_gt, args.train_lr, split='val', val_fraction=0.1, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)

    model = LightSRNet().to(device)
    criterion = CombinedLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_val_psnr = -1e9
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (degraded, target) in enumerate(train_loader):
            degraded = degraded.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(degraded)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / max(1, len(train_loader))
        history['train_loss'].append(train_loss)

        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        num_batches = 0
        val_pred_list = []
        val_target_list = []
        val_degraded_list = []

        with torch.no_grad():
            for degraded, target in val_loader:
                degraded = degraded.to(device)
                target = target.to(device)
                pred = model(degraded)

                loss = criterion(pred, target)
                val_loss += loss.item()
                val_psnr += psnr(pred, target)
                val_ssim += ssim(pred, target)

                num_batches += 1
                val_pred_list.append(pred.cpu())
                val_target_list.append(target.cpu())
                val_degraded_list.append(degraded.cpu())

        avg_loss = val_loss / max(1, num_batches)
        avg_psnr = val_psnr / max(1, num_batches)
        avg_ssim = val_ssim / max(1, num_batches)

        history['val_loss'].append(avg_loss)
        history['val_psnr'].append(avg_psnr)
        history['val_ssim'].append(avg_ssim)

        print(f'Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={avg_loss:.4f}, val_psnr={avg_psnr:.2f}, val_ssim={avg_ssim:.4f}')

        if avg_psnr > best_val_psnr:
            best_val_psnr = avg_psnr
            best_state = {k: v.clone() if hasattr(v, 'clone') else v for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint_dir / 'best_model.pth')
            print('Saved best checkpoint.')

        if num_batches > 0:
            val_pred = torch.cat(val_pred_list, dim=0)
            val_target = torch.cat(val_target_list, dim=0)
            val_degraded = torch.cat(val_degraded_list, dim=0)
            save_validation_grid(results_dir / f'val_grid_epoch_{epoch+1}.png', val_degraded, val_pred, val_target, n=4)

    metrics = {
        'best_val_psnr': float(best_val_psnr),
        'final_history': history,
        'device': str(device),
        'parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    save_json(results_dir / 'validation_summary.json', metrics)

    # Benchmark speed on CPU
    sample = torch.randn(1, 1, 128, 128)
    sample = sample.to(device)
    inf_time = benchmark_inference(model, sample, device=device, repeats=20)
    print(f'Average inference time per image: {inf_time:.4f} s')
    save_json(results_dir / 'inference_benchmark.json', {'cpu_or_gpu_seconds_per_image': float(inf_time), 'device': str(device)})

    print('Training complete.')


if __name__ == '__main__':
    main()
