import argparse
from pathlib import Path

import numpy as np
import torch

from src.dataset import normalize_input_image
from src.model import LightSRNet


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference for repaired semiconductor images.')
    parser.add_argument('--input-path', type=str, default=None, help='Single noisy .npy image to restore.')
    parser.add_argument('--input-dir', type=str, default=None, help='Folder of noisy .npy images to restore in batch.')
    parser.add_argument('--output-path', type=str, default=None, help='Output path for a single restored image.')
    parser.add_argument('--output-dir', type=str, default=None, help='Output folder for batch restoration results.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/demo_model.pth')
    return parser.parse_args()


def infer_single(model, image_path, device):
    degraded = np.load(image_path).astype(np.float32)
    degraded = normalize_input_image(degraded)
    degraded = torch.from_numpy(degraded).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        restored = model(degraded)

    restored = restored[0, 0].cpu().numpy()
    restored = np.clip(restored, 0.0, 1.0)
    return restored.astype(np.float32)


def main():
    args = parse_args()
    if args.input_path is None and args.input_dir is None:
        raise ValueError('Provide either --input-path or --input-dir.')
    if args.input_path is not None and args.output_path is None:
        raise ValueError('Provide --output-path when using --input-path.')
    if args.input_dir is not None and args.output_dir is None:
        raise ValueError('Provide --output-dir when using --input-dir.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = LightSRNet().to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    if args.input_path is not None:
        restored = infer_single(model, args.input_path, device)
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, restored)
        print(f'Saved restored image to {out_path}')
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob('*.npy'))
    if not files:
        raise FileNotFoundError(f'No .npy files found in {input_dir}')

    for image_path in files:
        restored = infer_single(model, str(image_path), device)
        out_path = output_dir / image_path.name
        np.save(out_path, restored)

    print(f'Processed {len(files)} files from {input_dir}')
    print(f'Saved restored outputs to {output_dir}')


if __name__ == '__main__':
    main()
