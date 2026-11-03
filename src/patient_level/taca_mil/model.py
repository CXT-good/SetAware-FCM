from __future__ import annotations

import torch
import torch.nn as nn

from common.set_attention import SetTransformerEncoder
from mae.model import MAEEncoder
from common.attention import GatedAttention


class PatientLevelTubeSetTransformerMIL(nn.Module):
    """
    Patient-level stage2 model:
    frozen MAE -> marker gated attention -> per-tube SetTransformer -> patient MIL.
    SetTransformer is applied independently inside each tube.
    """

    def __init__(
        self,
        encoder: MAEEncoder,
        *,
        num_classes: int,
        set_hidden_dim: int = 128,
        set_num_heads: int = 4,
        set_num_inds: int = 16,
        set_num_layers: int = 2,
        attention_hidden_dim: int = 128,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.1,
        ln: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.marker_attention = GatedAttention(encoder.cfg.d_model, attention_hidden_dim)
        self.tube_set_transformer = SetTransformerEncoder(
            dim_input=encoder.cfg.d_model,
            dim_hidden=set_hidden_dim,
            num_heads=set_num_heads,
            num_inds=set_num_inds,
            num_layers=set_num_layers,
            ln=ln,
        )
        self.cell_attention = GatedAttention(set_hidden_dim, attention_hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(set_hidden_dim),
            nn.Linear(set_hidden_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True) -> "PatientLevelTubeSetTransformerMIL":
        super().train(mode)
        self.encoder.eval()
        return self

    def encode_tube(
        self,
        marker_ids: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 2:
            raise ValueError("values must be [num_cells, num_markers]")
        marker_ids_batch = marker_ids.unsqueeze(0).expand(values.shape[0], -1)
        with torch.no_grad():
            marker_features = self.encoder(marker_ids_batch, values, mask=None)
        static_cell_features, marker_weights = self.marker_attention(marker_features, dim=1)
        dynamic_cell_features = self.tube_set_transformer(static_cell_features.unsqueeze(0)).squeeze(0)
        return dynamic_cell_features, marker_weights, static_cell_features

    def forward(self, tubes: list[dict], *, return_attention: bool = False) -> dict:
        dynamic_features_list: list[torch.Tensor] = []
        static_features_list: list[torch.Tensor] = []
        metadata: list[dict] = []
        marker_attention_by_tube: list[torch.Tensor] = []

        for tube_idx, tube in enumerate(tubes):
            dynamic_features, marker_weights, static_features = self.encode_tube(tube["marker_ids"], tube["values"])
            dynamic_features_list.append(dynamic_features)
            static_features_list.append(static_features)
            marker_attention_by_tube.append(marker_weights)
            metadata.append(
                {
                    "tube_idx": tube_idx,
                    "num_cells": int(dynamic_features.shape[0]),
                    "csv_path": tube["csv_path"],
                    "tube_type": tube["tube_type"],
                    "marker_names": tube["marker_names"],
                    "row_ids": tube["row_ids"],
                    "values": tube["values"],
                }
            )

        if not dynamic_features_list:
            raise ValueError("patient has no tubes")
        dynamic_cell_bag = torch.cat(dynamic_features_list, dim=0)
        patient_repr, cell_weights = self.cell_attention(dynamic_cell_bag, dim=0)
        logits = self.classifier(patient_repr.unsqueeze(0)).squeeze(0)
        out = {"logits": logits}
        if return_attention:
            out.update(
                {
                    "patient_repr": patient_repr,
                    "dynamic_cell_features": dynamic_cell_bag,
                    "static_cell_features": torch.cat(static_features_list, dim=0),
                    "cell_attention": cell_weights,
                    "marker_attention_by_tube": marker_attention_by_tube,
                    "metadata": metadata,
                }
            )
        return out
