"""Small MLP heads — the only trainable parts (encoders stay frozen)."""
from __future__ import annotations

from typing import Sequence

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int | Sequence[int], out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        hs = [hidden] if isinstance(hidden, int) else list(hidden)
        layers: list[nn.Module] = []
        d = in_dim
        for h in hs:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
