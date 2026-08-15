import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import SemiconductorPairDataset
from src.model import LightSRNet
from src.utils import CombinedLoss, psnr, save_validation_grid, ssim


def main():
    train_gt = r'C:\Users\Welcome\Downloads\train\train\GT'
    train_lr = r'C:\Users\Welcome\Downloads\train\train\NoisyLR'

    train_ds = SemiconductorPairDataset(train_gt, train_lr, split='train', val_fraction=0.1, augment=True)
    val_ds = SemiconductorPairDataset(train_gt, train_lr, split='val', val_fraction=0.1, augment=False)

    train_ds = Subset(train_ds, list(range(min(64, len(train_ds)))))
    val_ds = Subset(val_ds, list(range(min(8, len(val_ds)))))

    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    device = torch.device('cpu')
    model = LightSRNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = CombinedLoss().to(device)

    train_losses = []
    model.train()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))

    model.eval()
    val_pred = []
    val_target = []
    val_degraded = []
    val_psnr = []
    val_ssim = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            val_pred.append(pred.cpu())
            val_target.append(y.cpu())
            val_degraded.append(x.cpu())
            val_psnr.append(psnr(pred, y))
            val_ssim.append(ssim(pred, y))

    val_pred = torch.cat(val_pred, dim=0)
    val_target = torch.cat(val_target, dim=0)
    val_degraded = torch.cat(val_degraded, dim=0)

    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True, parents=True)
    save_validation_grid(out_dir / 'val_grid_demo.png', val_degraded, val_pred, val_target, n=4)

    metrics = {
        'train_loss': float(np.mean(train_losses)),
        'val_psnr': float(np.mean(val_psnr)),
        'val_ssim': float(np.mean(val_ssim)),
        'num_train_samples': len(train_ds),
        'num_val_samples': len(val_ds),
        'device': str(device),
    }
    with open(out_dir / 'validation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    torch.save(model.state_dict(), 'checkpoints/demo_model.pth')

    # Benchmark speed for a single image
    sample = torch.randn(1, 1, 128, 128, device=device)
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(20):
            t0 = __import__('time').perf_counter()
            _ = model(sample)
            times.append(__import__('time').perf_counter() - t0)

    with open(out_dir / 'inference_benchmark.json', 'w', encoding='utf-8') as f:
        json.dump({'avg_seconds_per_image': float(np.mean(times)), 'device': str(device), 'repeats': 20}, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print('Files saved:')
    print(' -', out_dir / 'val_grid_demo.png')
    print(' -', out_dir / 'validation_summary.json')
    print(' -', out_dir / 'inference_benchmark.json')
    print(' - checkpoints/demo_model.pth')


if __name__ == '__main__':
    main()
