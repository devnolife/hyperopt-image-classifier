"""Custom CNN model untuk CIFAR-10 (configurable)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    """3 ConvBlock + FC. base_filters dan dropout bisa dikonfigurasi."""

    def __init__(self, num_classes: int = 10, base_filters: int = 32, dropout: float = 0.3):
        super().__init__()
        f1 = base_filters
        f2 = base_filters * 2
        f3 = base_filters * 4
        self.features = nn.Sequential(
            ConvBlock(3, f1, dropout),      # 32 -> 16
            ConvBlock(f1, f2, dropout),     # 16 -> 8
            ConvBlock(f2, f3, dropout),     # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f3 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_optimizer(name: str, params, lr: float, weight_decay: float = 5e-4):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")
