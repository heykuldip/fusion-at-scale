"""Models for the fusion comparison: small temporal convolutions, one per stream, matched in size.

  joints, imu   each stream's motion (temporal encoder) and posture (embedding), then a two-layer head
                (single-stream baselines; late fusion averages their probabilities)
  intermediate  both streams' encoders and embeddings, joined before one head of the two single heads' size
  position      an encoder over the hip's path in camera coordinates (negative control)
  mask          a small MLP over which channels are measured (negative control)
"""
import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """(batch, time, features) -> (batch, 2 x out): three convolutions with growing dilation, then mean and max
    over time."""

    def __init__(self, in_dim, width=64, out=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, width, 5, padding=2), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(width, width, 5, padding=4, dilation=2), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(width, out, 5, padding=8, dilation=4), nn.BatchNorm1d(out), nn.GELU())
        self.dim = 2 * out

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return torch.cat([h.mean(2), h.amax(2)], 1)


def head(in_dim, classes, hidden=128, dropout=0.3):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, classes))


class Single(nn.Module):
    """One stream: its motion through the temporal encoder, its posture (if any) through a small embedding, joined
    before the head."""

    def __init__(self, in_dim, classes, static_dim=0):
        super().__init__()
        self.enc = TemporalEncoder(in_dim)
        self.static = nn.Sequential(nn.Linear(static_dim, 64), nn.GELU()) if static_dim else None
        self.head = head(self.enc.dim + (64 if static_dim else 0), classes)

    def forward(self, x, s=None):
        h = self.enc(x)
        if self.static is not None:
            h = torch.cat([h, self.static(s)], 1)
        return self.head(h)


class Intermediate(nn.Module):
    """Both streams' encoders and posture embeddings, joined before one head. Its input is the two single-stream
    heads' inputs side by side, and its hidden layer has as many weights as theirs together."""

    def __init__(self, joints_dim, joints_static, imu_dim, imu_static, classes):
        super().__init__()
        self.enc_j, self.enc_i = TemporalEncoder(joints_dim), TemporalEncoder(imu_dim)
        self.st_j = nn.Sequential(nn.Linear(joints_static, 64), nn.GELU())
        self.st_i = nn.Sequential(nn.Linear(imu_static, 64), nn.GELU())
        self.head = head(self.enc_j.dim + self.enc_i.dim + 128, classes)   # 640 x 128 = 2 x (320 x 128)

    def forward(self, xj, sj, xi, si):
        return self.head(torch.cat([self.enc_j(xj), self.st_j(sj), self.enc_i(xi), self.st_i(si)], 1))


class VectorMLP(nn.Module):
    def __init__(self, in_dim, classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, classes))

    def forward(self, x):
        return self.net(x)


def parameters(model):
    return sum(p.numel() for p in model.parameters())


__all__ = ['TemporalEncoder', 'Single', 'Intermediate', 'VectorMLP', 'parameters']
