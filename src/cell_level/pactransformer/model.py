"""PACTransformer population-context classifier."""

import torch
import torch.nn as nn

from common.set_attention import ISAB, MAB, SAB, SetTransformerEncoder


class SetTransformer(nn.Module):
    """
    完整的Set Transformer模型
    用于阶段二的分布建模分类
    SetTransformer(X) = SetTransformerEncoder(X)
    """
    def __init__(
        self,
        dim_input: int = 32, # 输入维度 （默认32）
        dim_hidden: int = 32, # 隐藏层维度 （默认32）
        num_heads: int = 4, # 注意力头数 （默认4）
        num_inds: int = 16, # 诱导点数量 （默认16）
        num_layers: int = 3, # ISAB层数 （默认3）
        num_classes: int = 2, # 分类类别数（应该根据CSV文件中的标签自动检测）
        ln: bool = True, # 是否使用层归一化 （默认True）
        dropout: float = 0.1, # Dropout率 （默认0.1）
    ):
        super(SetTransformer, self).__init__() # 初始化
        
        self.num_classes = num_classes
        
        self.encoder = SetTransformerEncoder(
            dim_input=dim_input, # 输入维度
            dim_hidden=dim_hidden, # 隐藏层维度
            num_heads=num_heads, # 注意力头数
            num_inds=num_inds, # 诱导点数量
            num_layers=num_layers, # ISAB层数
            ln=ln # 是否使用层归一化
        )
        
        # 分类头：使用简单的线性层（更适合可视化）
        self.classifier = nn.Linear(dim_hidden, self.num_classes)
        
        # Dropout层：在分类前应用，用于正则化
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X): # 前向传播
        """
        Args:
            X: (batch_size, N, dim_input) 细胞嵌入集合
        Returns:
            (batch_size, N, num_classes) 每个细胞的分类logits
        """
        # 编码
        encoded = self.encoder(X)  # (batch_size, N, dim_hidden) 编码后的细胞表示
        
        # Dropout
        encoded = self.dropout(encoded) # Dropout
        
        # 分类
        logits = self.classifier(encoded)  # (batch_size, N, num_classes) 分类logits
        
        return logits  # (batch_size, N, num_classes)
