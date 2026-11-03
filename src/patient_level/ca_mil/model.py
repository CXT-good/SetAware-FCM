from __future__ import annotations

import torch
import torch.nn as nn

from mae.model import MAEConfig, MAEEncoder
from common.attention import GatedAttention




class PatientLevelMIL(nn.Module):
    def __init__(
        self,
        encoder: MAEEncoder,
        *,
        num_classes: int,
        attention_hidden_dim: int = 128,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.marker_attention = GatedAttention(encoder.cfg.d_model, attention_hidden_dim)
        self.cell_attention = GatedAttention(encoder.cfg.d_model, attention_hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder.cfg.d_model),
            nn.Linear(encoder.cfg.d_model, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True) -> "PatientLevelMIL":
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
        cell_features, marker_weights = self.marker_attention(marker_features, dim=1)
        return cell_features, marker_weights, marker_features

    def forward(self, tubes: list[dict], *, return_attention: bool = False) -> dict:
        cell_features_list: list[torch.Tensor] = []
        metadata: list[dict] = []
        marker_attention_by_tube: list[torch.Tensor] = []

        for tube_idx, tube in enumerate(tubes):
            cell_features, marker_weights, _ = self.encode_tube(tube["marker_ids"], tube["values"])
            cell_features_list.append(cell_features)
            marker_attention_by_tube.append(marker_weights)
            metadata.append(
                {
                    "tube_idx": tube_idx,
                    "num_cells": int(cell_features.shape[0]),
                    "csv_path": tube["csv_path"],
                    "tube_type": tube["tube_type"],
                    "marker_names": tube["marker_names"],
                    "row_ids": tube["row_ids"],
                    "values": tube["values"],
                }
            )

        if not cell_features_list:
            raise ValueError("patient has no tubes")
        cell_bag = torch.cat(cell_features_list, dim=0)
        patient_repr, cell_weights = self.cell_attention(cell_bag, dim=0)
        logits = self.classifier(patient_repr.unsqueeze(0)).squeeze(0)
        out = {"logits": logits}
        if return_attention:
            out.update(
                {
                    "patient_repr": patient_repr,
                    "cell_features": cell_bag,
                    "cell_attention": cell_weights,
                    "marker_attention_by_tube": marker_attention_by_tube,
                    "metadata": metadata,
                }
            )
        return out


def build_encoder_from_checkpoint(
    checkpoint_path: str,
    *,
    num_markers: int,
    seq_len: int,
    d_model: int,
    encoder_layers: int,
    decoder_layers: int,
    n_heads: int,
    d_ff: int,
    dropout: float,
    device: torch.device,
) -> MAEEncoder:
    cfg = MAEConfig(
        num_markers=num_markers,
        seq_len=seq_len,
        d_model=d_model,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
    )
    encoder = MAEEncoder(cfg).to(device)
    if checkpoint_path:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload
        if isinstance(payload, dict):
            state = payload.get("encoder_state_dict") or payload.get("state_dict") or payload
        if isinstance(state, dict) and state and all(k.startswith("encoder.") for k in state.keys()):
            state = {k[len("encoder."):]: v for k, v in state.items()}
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(f"loaded encoder: missing={len(missing)} unexpected={len(unexpected)}")
    return encoder
