"""Shared network building blocks for world-model components."""

from __future__ import annotations

import torch.nn as nn


def mlp(in_dim: int, hidden: int, out_dim: int, depth: int = 2, dropout: float = 0.0) -> nn.Module:
    """Build a simple feed-forward MLP.

    Args:
        in_dim: Input feature size.
        hidden: Hidden layer width.
        out_dim: Output feature size.
        depth: Total number of linear layers (≥ 1).
        dropout: Dropout probability between hidden layers (0 = disabled).
    """
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)
