import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_dataset_dirs(gt_dir=None, lr_dir=None):
    """Return dataset folders, creating a tiny demo dataset if no real data exists."""
    root = Path(__file__).resolve().parent.parent
    default_gt = Path(gt_dir) if gt_dir else root / 'data' / 'train' / 'GT'
    default_lr = Path(lr_dir) if lr_dir else root / 'data' / 'train' / 'NoisyLR'

    gt_dir_path = Path(default_gt)
    lr_dir_path = Path(default_lr)

    if not gt_dir_path.exists() or not lr_dir_path.exists() or not any(gt_dir_path.glob('*.npy')) or not any(lr_dir_path.glob('*.npy')):
        gt_dir_path.mkdir(parents=True, exist_ok=True)
        lr_dir_path.mkdir(parents=True, exist_ok=True)
        if not any(gt_dir_path.glob('*.npy')) or not any(lr_dir_path.glob('*.npy')):
            _create_demo_dataset(gt_dir_path, lr_dir_path)

    return gt_dir_path, lr_dir_path


def _create_demo_dataset(gt_dir: Path, lr_dir: Path, count: int = 32):
    rng = np.random.default_rng(42)
    for idx in range(count):
        y, x = np.mgrid[0:128, 0:128]
        stripes = (y % 32) / 32.0
        grid = np.sin((x + idx * 7) / 8.0) * 0.2 + np.cos((y + idx * 11) / 10.0) * 0.1
        target = np.clip(0.35 + stripes * 0.5 + grid, 0.0, 1.0)
        lines = np.zeros((128, 128), dtype=np.float32)
        for pos in range(0, 128, 16):
            lines[:, pos:pos + 3] = 0.8
        target = np.clip(target * 0.85 + lines * 0.15, 0.0, 1.0)

        target_256 = np.zeros((256, 256), dtype=np.float32)
        for yy in range(0, 256, 2):
            for xx in range(0, 256, 2):
                target_256[yy:yy + 2, xx:xx + 2] = target[yy // 2, xx // 2]

        noise = rng.normal(0.0, 0.08, size=target.shape).astype(np.float32)
        degraded = np.clip(target + noise, 0.0, 1.0).astype(np.float32)
        np.save(gt_dir / f'{idx:06d}.npy', target_256.astype(np.float32))
        np.save(lr_dir / f'{idx:06d}.npy', degraded.astype(np.float32))


def normalize_input_image(image: np.ndarray) -> np.ndarray:
    """Clip and scale degraded inputs to a stable grayscale range.

    The noisy input can exceed the GT range due to speckle noise. We clip to
    [0, 1.5] and then scale by 1.5 so the network sees a compact [0, 1] range
    while preserving the artifact structure.
    """
    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 0.0, 1.5)
    image = image / 1.5
    return image.astype(np.float32)


def normalize_target_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32)


class SemiconductorPairDataset(Dataset):
    def __init__(self, gt_dir, lr_dir, split='train', val_fraction=0.1, seed=42, augment=True):
        self.gt_dir = Path(gt_dir)
        self.lr_dir = Path(lr_dir)
        self.split = split
        self.augment = augment
        self.seed = seed

        self.samples = []
        for gt_path in sorted(self.gt_dir.glob('*.npy')):
            stem = gt_path.stem
            lr_path = self.lr_dir / f'{stem}.npy'
            if lr_path.exists():
                self.samples.append((lr_path, gt_path))

        rng = random.Random(seed)
        rng.shuffle(self.samples)
        split_idx = max(1, int(len(self.samples) * (1.0 - val_fraction)))

        if split == 'train':
            self.samples = self.samples[:split_idx]
        elif split == 'val':
            self.samples = self.samples[split_idx:]
        else:
            raise ValueError(f'Unsupported split: {split}')

    def __len__(self):
        return len(self.samples)

    def _augment(self, degraded: np.ndarray, target: np.ndarray):
        if not self.augment:
            return degraded, target

        if random.random() < 0.5:
            degraded = np.flip(degraded, axis=1)
            target = np.flip(target, axis=1)
        if random.random() < 0.5:
            degraded = np.flip(degraded, axis=0)
            target = np.flip(target, axis=0)

        rot_k = random.choice([0, 1, 2, 3])
        if rot_k:
            degraded = np.rot90(degraded, k=rot_k, axes=(0, 1))
            target = np.rot90(target, k=rot_k, axes=(0, 1))

        return degraded, target

    def __getitem__(self, idx):
        lr_path, gt_path = self.samples[idx]
        degraded = np.load(lr_path)
        target = np.load(gt_path)

        degraded = np.asarray(degraded, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        degraded, target = self._augment(degraded, target)

        degraded = normalize_input_image(degraded)
        target = normalize_target_image(target)

        degraded = torch.from_numpy(degraded).unsqueeze(0)
        target = torch.from_numpy(target).unsqueeze(0)
        return degraded, target


def inspect_dataset(gt_dir, lr_dir):
    gt_files = sorted(Path(gt_dir).glob('*.npy'))
    lr_files = sorted(Path(lr_dir).glob('*.npy'))
    print(f'GT files: {len(gt_files)}')
    print(f'LR files: {len(lr_files)}')
    print(f'First GT sample: {gt_files[0]}')
    print(f'First LR sample: {lr_files[0]}')

    gt_sample = np.load(gt_files[0])
    lr_sample = np.load(lr_files[0])
    print(f'GT shape={gt_sample.shape}, dtype={gt_sample.dtype}, min={gt_sample.min():.4f}, max={gt_sample.max():.4f}')
    print(f'LR shape={lr_sample.shape}, dtype={lr_sample.dtype}, min={lr_sample.min():.4f}, max={lr_sample.max():.4f}')
    print(f'Pair count match: {len(gt_files) == len(lr_files)}')

    if len(gt_files) == len(lr_files):
        mismatched = [
            (gt.name, lr.name)
            for gt, lr in zip(sorted(gt_files), sorted(lr_files))
            if gt.stem != lr.stem
        ]
        if mismatched:
            print('First mismatches:', mismatched[:5])
