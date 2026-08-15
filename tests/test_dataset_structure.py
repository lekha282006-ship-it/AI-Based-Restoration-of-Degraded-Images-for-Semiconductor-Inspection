import os
from pathlib import Path

from src.dataset import resolve_dataset_dirs


def test_dataset_pairing():
    train_gt, train_lr = resolve_dataset_dirs()
    assert train_gt.exists(), f"Missing GT dir: {train_gt}"
    assert train_lr.exists(), f"Missing LR dir: {train_lr}"
    gt_files = sorted(p.name for p in train_gt.glob("*.npy"))
    lr_files = sorted(p.name for p in train_lr.glob("*.npy"))
    assert len(gt_files) > 0
    assert len(lr_files) > 0
    assert len(gt_files) == len(lr_files)
    assert gt_files[:3] == lr_files[:3]


if __name__ == "__main__":
    test_dataset_pairing()
    print("dataset pairing ok")
