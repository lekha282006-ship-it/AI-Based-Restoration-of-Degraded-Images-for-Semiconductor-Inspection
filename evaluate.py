import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lpips import LPIPS
from skimage.metrics import structural_similarity

from src.dataset import normalize_input_image
from src.model import LightSRNet


def mse_to_psnr(mse):
    if mse <= 1e-12:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def compute_metrics(pred, target):
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = float(torch.mean((pred - target) ** 2).item())
    psnr = mse_to_psnr(mse)
    ssim_value = structural_similarity(
        target.detach().cpu().numpy()[0, 0],
        pred.detach().cpu().numpy()[0, 0],
        data_range=1.0,
    )
    return psnr, ssim_value


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a restored semiconductor image model on a folder of paired test inputs and GT outputs.')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory with noisy 128x128 .npy files')
    parser.add_argument('--gt-dir', type=str, required=True, help='Directory with clean 256x256 .npy targets')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/demo_model.pth')
    parser.add_argument('--output-dir', type=str, default='results/evaluation')
    parser.add_argument('--max-samples', type=int, default=None, help='Optional cap for how many paired files to evaluate; useful for a quick validation pass.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightSRNet().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    lpips_model = LPIPS(net='alex').to(device)
    lpips_model.eval()

    input_dir = Path(args.input_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p.name for p in input_dir.glob('*.npy'))
    if args.max_samples is not None:
        files = files[:max(0, args.max_samples)]

    metrics = {'psnr': [], 'ssim': [], 'lpips': []}

    for name in files:
        gt_path = gt_dir / name
        if not gt_path.exists():
            continue

        noisy = np.load(input_dir / name).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        noisy_t = torch.from_numpy(normalize_input_image(noisy)).unsqueeze(0).unsqueeze(0).to(device)
        gt_t = torch.from_numpy(np.clip(gt, 0.0, 1.0)).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            restored = model(noisy_t)
            pred_np = restored.cpu().clamp(0.0, 1.0)
            gt_np = gt_t.cpu().clamp(0.0, 1.0)

        psnr_value, ssim_value = compute_metrics(restored, gt_t)
        lpips_value = float(lpips_model(pred_np, gt_np).mean().item())

        metrics['psnr'].append(psnr_value)
        metrics['ssim'].append(ssim_value)
        metrics['lpips'].append(lpips_value)

        out_path = output_dir / name
        np.save(out_path, pred_np[0, 0].numpy().astype(np.float32))

    if not metrics['psnr']:
        raise FileNotFoundError(f'No matching .npy pairs were found between {input_dir} and {gt_dir}.')

    summary = {
        'count': len(metrics['psnr']),
        'psnr_mean': float(np.mean(metrics['psnr'])),
        'ssim_mean': float(np.mean(metrics['ssim'])),
        'lpips_mean': float(np.mean(metrics['lpips'])),
        'psnr_max': float(np.max(metrics['psnr'])),
        'ssim_max': float(np.max(metrics['ssim'])),
        'lpips_min': float(np.min(metrics['lpips'])),
    }

    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f'Wrote restored outputs to {output_dir}')


if __name__ == '__main__':
    main()
