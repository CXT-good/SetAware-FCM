"""Shared marker/cell gated-attention pooling."""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.v = nn.Linear(input_dim, hidden_dim)
        self.u = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))).squeeze(-1)
        weights = torch.softmax(scores, dim=dim)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=dim)
        return pooled, weights
