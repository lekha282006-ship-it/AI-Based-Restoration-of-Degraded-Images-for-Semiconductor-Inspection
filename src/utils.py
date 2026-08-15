import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity


def psnr(pred, target):
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return float('inf')
    return 10.0 * torch.log10(1.0 / mse).item()


def ssim(pred, target):
    pred_np = pred.detach().cpu().numpy()[0, 0]
    target_np = target.detach().cpu().numpy()[0, 0]
    return structural_similarity(target_np, pred_np, data_range=1.0)


def gradient_loss(pred, target):
    diff_x = pred[..., :, 1:] - pred[..., :, :-1]
    diff_y = pred[..., 1:, :] - pred[..., :-1, :]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    loss_x = torch.mean(torch.abs(diff_x - target_x))
    loss_y = torch.mean(torch.abs(diff_y - target_y))
    return 0.5 * (loss_x + loss_y)


class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        eps = 1e-3
        charbonnier = torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))
        grad = gradient_loss(pred, target)
        return l1_loss + 0.5 * charbonnier + 0.1 * grad


def save_validation_grid(path, degraded, pred, target, n=4):
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for i in range(n):
        d = degraded[i].detach().cpu().numpy()[0]
        p = pred[i].detach().cpu().numpy()[0]
        t = target[i].detach().cpu().numpy()[0]

        axes[i, 0].imshow(d, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Degraded')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(p, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Restored')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(t, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('GT')
        axes[i, 2].axis('off')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def benchmark_inference(model, input_tensor, device, repeats=20):
    model.eval()
    timings = []
    with torch.no_grad():
        for _ in range(repeats):
            start = __import__('time').perf_counter()
            _ = model(input_tensor.to(device))
            torch.cuda.synchronize() if hasattr(torch, 'cuda') and torch.cuda.is_available() else None
            timings.append(__import__('time').perf_counter() - start)
    return float(np.mean(timings))
