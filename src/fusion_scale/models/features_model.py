"""The features model: a small Transformer encoder over time, then a classifier.

width and depth set its size, and so how much compute a step has against the gradient all-reduce. The default
(width 256, depth 4) has about 3 million parameters, small enough to be overhead-bound; a larger setting such as
width 1024, depth 8 (about 100 million) makes communication visible.
"""
import torch
import torch.nn as nn


class FeaturesModel(nn.Module):
    def __init__(self, f: int = 64, t: int = 64, width: int = 256, depth: int = 4, heads: int = 4, classes: int = 13):
        super().__init__()
        self.inp = nn.Linear(f, width)
        self.pos = nn.Parameter(torch.zeros(1, t, width))
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=4 * width, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.head = nn.Linear(width, classes)

    def forward(self, x):
        h = self.encoder(self.inp(x) + self.pos)
        return self.head(h.mean(dim=1))
