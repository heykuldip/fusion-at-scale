"""Synthetic stand-ins for windows of pose and IMU features, for the scaling benchmark.

Each sample is a (T, F) float tensor with a class label. The values are random but fixed by the seed, and held in
memory, so loading costs what an in-memory feature dataset would: indexing, collating and copying to the GPU. The
labels depend on the data, so the model has something to learn and the loss falls.
"""
import torch
from torch.utils.data import Dataset


class SyntheticFeatures(Dataset):
    def __init__(self, n: int = 50_000, t: int = 64, f: int = 64, classes: int = 13, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, t, f, generator=g)
        w = torch.randn(f, classes, generator=g)
        self.y = (self.x.mean(dim=1) @ w).argmax(dim=1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
