"""CIFAR-10 data loaders."""
from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, eval_tf


def get_dataloaders(
    root: str | Path = "./data",
    batch_size: int = 128,
    val_size: int = 5000,
    num_workers: int = 2,
    seed: int = 42,
    train_subset: int | None = None,
):
    """Return (train_loader, val_loader, test_loader, classes).

    Jika ``train_subset`` diberikan, training set dibatasi ke N sampel pertama
    (setelah shuffle reproducible) untuk mempercepat eksperimen HPO.
    """
    train_tf, eval_tf = _build_transforms()

    full_train = datasets.CIFAR10(root=str(root), train=True, download=True, transform=train_tf)
    # dataset kembar untuk val (tanpa augmentasi)
    full_train_eval = datasets.CIFAR10(root=str(root), train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(root=str(root), train=False, download=True, transform=eval_tf)

    n = len(full_train)
    train_n = n - val_size
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx = perm[:train_n]
    val_idx = perm[train_n:]

    if train_subset is not None and train_subset < len(train_idx):
        train_idx = train_idx[:train_subset]

    train_subset_ds = Subset(full_train, train_idx)
    val_subset = Subset(full_train_eval, val_idx)

    train_loader = DataLoader(
        train_subset_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_subset, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader, full_train.classes
