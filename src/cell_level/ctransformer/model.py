"""
阶段一：静态细胞分类Transformer
使用指标嵌入和数值嵌入处理不同标记物组合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MarkerEmbedding(nn.Module):
    """
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


class Stage1Transformer(nn.Module):
    """
    阶段一：静态细胞分类Transformer
    
    注意：
    - num_classes 参数应该根据CSV文件中的标签自动检测，不应该硬编码
    - 在 train_stage1.py 中会自动检测数据中的类别数量
    - 默认值2仅用于函数参数的兼容性，实际使用时会被自动检测的值覆盖
    """
    def __init__(
        self,
        markers: list = None, # 所有标记物名称对应的整数索引列表，如果为None则使用索引
        embed_dim: int = 32, # 嵌入维度 
        num_heads: int = 4, # 注意力头数
        num_layers: int = 3, # Transformer编码器层数
        dim_feedforward: int = 128, # 前馈网络维度
        dropout: float = 0.1, # Dropout率
        num_classes: int = 2, # 分类类别数（注意：应该根据CSV文件中的标签自动检测，默认值2仅用于兼容性）
        value_embedding_mode: str = "shared",  # 保留参数以兼容旧 checkpoint/脚本；仅支持 shared
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        num_markers = len(markers)
        # 如果没有提供markers列表，则使用索引作为名称（向后兼容）
        if markers is None:
            markers = [str(i) for i in range(num_markers)]
        self.markers = markers
        
        if value_embedding_mode != "shared":
            raise ValueError(
                f"value_embedding_mode 仅支持 'shared'（共享 Linear(1->embed_dim)），当前为: {value_embedding_mode}"
            )
        self.value_embedding_mode = value_embedding_mode

        # 嵌入层
        self.marker_embedding = MarkerEmbedding(num_markers, embed_dim)
        self.value_embedding = ValueEmbedding(embed_dim)
        self.value_ln = nn.LayerNorm(embed_dim)
        # CLS Token（可学习的）
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads, # 注意力头数
            dim_feedforward=dim_feedforward, # 前馈网络维度
            dropout=dropout, # Dropout率
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # Transformer编码器 （num_layers层）
        
        # 分类头（简化版：直接映射，因为Transformer已经做了足够的特征提取）
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout) # Dropout层
    
    # 前向传播
    def forward(
        self,
        marker_values: torch.Tensor, # (batch_size, seq_len) 标记物表达值
        marker_indices: torch.LongTensor, # (seq_len,) 标记物索引（所有样本共享）
    ) -> torch.Tensor:
        """
        Args:
            marker_values: (batch_size, seq_len) 标记物表达值
            marker_indices: (seq_len,) 标记物索引（所有样本共享相同的标记物索引）
        Returns:
            (batch_size, num_classes) 分类logits
            (batch_size, embed_dim) 细胞嵌入向量
        """
        batch_size, seq_len = marker_values.shape
        
        # 将 marker_indices 扩展为 (batch_size, seq_len) 以便批量处理
        marker_indices_batch = marker_indices.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
        
        # 嵌入
        marker_emb = self.marker_embedding(marker_indices_batch)  # (batch_size, seq_len, embed_dim) 标记物嵌入向量
        value_emb = self.value_embedding(marker_values)  # (batch_size, seq_len, embed_dim)
        value_emb = self.value_ln(value_emb)  # (batch_size, seq_len, embed_dim) 数值嵌入向量归一化
        # 相加得到Token
        tokens = marker_emb + value_emb  # (batch_size, seq_len, embed_dim) 标记物嵌入向量和数值嵌入向量相加得到Token
        tokens = self.dropout(tokens)  # Dropout

        # 添加CLS Token
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim) CLS Token
        tokens = torch.cat([cls_token, tokens], dim=1)  # (batch_size, seq_len+1, embed_dim) 将CLS Token和Token拼接
        
        # Transformer编码
        encoded = self.transformer(tokens)  # (batch_size, seq_len+1, embed_dim) Transformer编码
        
        # 提取CLS Token的输出
        cls_output = encoded[:, 0, :]  # (batch_size, embed_dim) CLS Token的输出
        
        # 分类
        logits = self.classifier(cls_output)  # (batch_size, num_classes) 分类logits
        
        return logits, cls_output
    
    def extract_embeddings(
        self,
        marker_values: torch.Tensor, # (batch_size, seq_len) 标记物表达值
        marker_indices: torch.LongTensor, # (seq_len,) 标记物索引（所有样本共享）
    ) -> torch.Tensor:
        """
        仅提取嵌入向量，不进行分类
        """
        _, embeddings = self.forward(marker_values, marker_indices)  # 提取嵌入向量
        return embeddings  # 返回所有细胞的嵌入向量 (batch_size, embed_dim)

