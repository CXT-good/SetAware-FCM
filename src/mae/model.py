from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class MarkerEmbedding(nn.Module):
    """
    复用 FCM/models/stage1_transformer.py 的写法：
    将每个 marker id 映射为 d_model 维向量。
    """

    def __init__(self, num_markers: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_markers, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, marker_indices: torch.LongTensor) -> torch.Tensor:
        # marker_indices: [B, N]
        return self.embedding(marker_indices)


class ValueEmbedding(nn.Module):
    """
    共享数值嵌入层：
    所有 marker 共用同一个线性层 (1 -> d_model)。
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(1, embed_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, marker_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            marker_values: [B, N]
        Returns:
            [B, N, D]
        """
        if marker_values.ndim != 2:
            raise ValueError("marker_values 需要是 [B, N]")

        values = marker_values.unsqueeze(-1)  # [B,N,1]
        return self.linear(values)  # [B,N,D]


def build_transformer_encoder(
    *,
    num_layers: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    dropout: float,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_ff,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


@dataclass
class MAEConfig:
    num_markers: int
    seq_len: int
    d_model: int = 128
    encoder_layers: int = 6
    decoder_layers: int = 2
    n_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.1
    mask_ratio: float = 0.3


class MarkerValueEmbed(nn.Module):
    def __init__(self, *, num_markers: int, d_model: int) -> None:
        super().__init__()
        # 复用 Stage1 的 marker/value embedding 设计
        self.marker_emb = MarkerEmbedding(num_markers, d_model)
        self.value_emb = ValueEmbedding(d_model)
        self.value_ln = nn.LayerNorm(d_model)

        # 可学习的 value 掩码向量：只替换 value embedding，不替换 marker embedding
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def forward(
        self,
        marker_ids: torch.LongTensor,  # [B, N]
        values: torch.FloatTensor,  # [B, N]
        *,
        mask: Optional[torch.BoolTensor] = None,  # [B, N] True 表示该位置要“遮罩 value”
    ) -> torch.Tensor:
        # marker/value 输入都按 [B, N] 组织，N 是 marker token 数
        if marker_ids.ndim != 2 or values.ndim != 2:
            raise ValueError("marker_ids/values 需要是 [B, N]")
        if marker_ids.shape != values.shape:
            raise ValueError("marker_ids 与 values 的形状需要相同")

        m = self.marker_emb(marker_ids)  # [B,N,D]
        v = self.value_emb(values)  # [B,N,D]
        v = self.value_ln(v)
        if mask is not None:
            # MAE 关键点：遮罩只作用在 value 分支，marker 身份 embedding 始终保留。
            mask_tok = self.mask_token.view(1, 1, -1).expand_as(v)
            v = torch.where(mask.unsqueeze(-1), mask_tok, v)
        # token 融合：x_i = E_m(m_i) + E_v(v_i)
        return m + v


class MAEEncoder(nn.Module):
    def __init__(self, cfg: MAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = MarkerValueEmbed(num_markers=cfg.num_markers, d_model=cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.enc = build_transformer_encoder(
            num_layers=cfg.encoder_layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        marker_ids: torch.LongTensor,
        values: torch.FloatTensor,
        *,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        x = self.embed(marker_ids, values, mask=mask)
        x = self.drop(x)
        return self.enc(x)


class ValueReconstructionHead(nn.Module):
    """
    共享重建头：
    所有 marker 共用同一个线性层，将 hidden -> 标量 value。
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x 需要是 [B, N, D]")
        return self.linear(x).squeeze(-1)  # [B,N]


def sample_mask(
    *,
    batch_size: int,#批量大小
    seq_len: int,#序列长度
    mask_ratio: float,#遮罩比例
    device: torch.device,
) -> torch.BoolTensor:
    """
    随机采样 mask 索引集合 M：batch 内每个样本各自一套随机遮罩位置（无放回）。

    返回：
    - mask: [B, N] bool，True 表示该位置属于 M（需要遮罩 value，并在 loss 中计算）
    """
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in (0,1)")
    num_mask = max(1, int(round(seq_len * mask_ratio)))  # 计算需要遮罩的 token 数量
    # 每行一次随机排列，取前 num_mask 列 → 每样本互不相同的遮罩下标（向量化，无 for）
    noise = torch.rand(batch_size, seq_len, device=device)
    idx = noise.argsort(dim=1)[:, :num_mask]  # [B, num_mask]
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    batch_idx = torch.arange(batch_size, device=device)[:, None].expand(-1, num_mask)
    mask[batch_idx, idx] = True
    return mask


class MAEModel(nn.Module):
    def __init__(self, cfg: MAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = MAEEncoder(cfg)
        self.decoder = build_transformer_encoder(
            num_layers=cfg.decoder_layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
        )
        self.recon_head = ValueReconstructionHead(cfg.d_model)

    def forward(
        self,
        marker_ids: torch.LongTensor,
        values: torch.FloatTensor,
        *,
        mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        # 1) 先构建“包含 mask token 的初始特征”（完整序列）
        x_full = self.encoder.embed(marker_ids, values, mask=mask)  # [B,N,D]
        x_full = self.encoder.drop(x_full)

        # 2) 编码器只处理可见 token（未遮挡位置）
        visible_mask = ~mask  # [B,N]
        visible_counts = visible_mask.sum(dim=1)
        if not torch.all(visible_counts == visible_counts[0]):
            raise ValueError("当前实现要求每个样本的可见 token 数一致")
        visible_len = int(visible_counts[0].item())
        if visible_len <= 0:
            raise ValueError("可见 token 数必须 > 0")

        x_visible = x_full[visible_mask].view(x_full.shape[0], visible_len, x_full.shape[-1])  # [B,N_vis,D]
        enc_visible = self.encoder.enc(x_visible)  # [B,N_vis,D]

        # 3) 将“编码后的可见特征”与“被遮挡 token 的初始特征”拼接成完整序列送入解码器
        dec_in = x_full.clone()
        dec_in[visible_mask] = enc_visible.reshape(-1, enc_visible.shape[-1])
        dec = self.decoder(dec_in)

        pred = self.recon_head(dec)  # [B, N]
        # loss 侧通常只在 mask=True 的位置计算重建误差（mask-only loss）。
        return pred, mask


class CellClassifier(nn.Module):
    def __init__(self, encoder: MAEEncoder, *, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder.cfg.d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.head = nn.Linear(encoder.cfg.d_model, num_classes)

    def forward(self, marker_ids: torch.LongTensor, values: torch.FloatTensor) -> torch.Tensor:
        # 下游分类阶段：CLS 在输入端与 token 初始化表示拼接后，共同输入编码器。
        x = self.encoder.embed(marker_ids, values, mask=None)  # [B,N,D]
        x = self.encoder.drop(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)  # [B,1,D]
        x = torch.cat([cls, x], dim=1)  # [B,N+1,D]
        x = self.encoder.enc(x)
        cls_out = x[:, 0, :]  # [B,D]
        return self.head(cls_out)

