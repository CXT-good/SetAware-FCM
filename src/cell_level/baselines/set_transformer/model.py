"""Set Transformer-only classifier on raw marker vectors."""

import torch
import torch.nn as nn

from cell_level.pactransformer.model import SetTransformer


class StandaloneSetTransformer(nn.Module):
    """Raw marker vectors -> Set Transformer -> per-cell logits."""

    def __init__(
        self,
        *,
        dim_input: int,
        dim_hidden: int,
        num_heads: int,
        num_inds: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(dim_input)
        self.set_transformer = SetTransformer(
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            num_inds=num_inds,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.set_transformer(self.input_norm(x))
