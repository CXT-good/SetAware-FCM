"""
阶段二：基于Set Transformer的分布建模分类
参考提供的Set Transformer实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#导入multihead_attention
from torch.nn import MultiheadAttention


# 多头注意力块（MAB）
class MAB(nn.Module):
    """
    Multihead Attention Block (MAB)
    多头注意力块（MAB）
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V # 输出维度
        self.num_heads = num_heads # 注意力头数
        self.fc_q = nn.Linear(dim_Q, dim_V) # 查询映射
        self.fc_k = nn.Linear(dim_K, dim_V) # 键映射
        self.fc_v = nn.Linear(dim_K, dim_V) # 值映射
        if ln: # 是否使用层归一化
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V) # 第二层层归一化
        self.fc_o = nn.Linear(dim_V, dim_V) # 输出映射
        self.ln = ln # 是否使用层归一化

    def forward(self, Q, K):
        Q = self.fc_q(Q) # 查询映射
        K, V = self.fc_k(K), self.fc_v(K) # 键映射和值映射
        dim_split = self.dim_V // self.num_heads # 每个头的大小
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0) # 查询拼接
        K_ = torch.cat(K.split(dim_split, 2), dim=0) # 键拼接
        V_ = torch.cat(V.split(dim_split, 2), dim=0) # 值拼接
        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2) # 注意力权重
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2) # 输出拼接
        O = O if not self.ln or getattr(self, 'ln0', None) is None else self.ln0(O) # 第一层层归一化
        O = O + F.relu(self.fc_o(O)) # 激活函数
        O = O if not self.ln or getattr(self, 'ln1', None) is None else self.ln1(O) # 第二层层归一化
        return O # 输出

# 自注意力块（SAB）
class SAB(nn.Module):
    """
    Set Attention Block (SAB)
    自注意力块（SAB）
    SAB(X) := MAB(X, X)
    """
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln) # 多头注意力块

    def forward(self, X):
        return self.mab(X, X) # 多头注意力块 （X, X）   

# 诱导点注意力块（ISAB）
class ISAB(nn.Module):
    """
    Induced Set Attention Block (ISAB)
    使用诱导点降低计算复杂度从O(n^2)到O(nm)
    诱导点注意力块（ISAB）
    ISAB(X) = MAB(X, H) where H = MAB(I, X)
    """
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out)) # 诱导点
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln) # 多头注意力块
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln) # 多头注意力块

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H) # 多头注意力块 （X, H）  

# Set Transformer编码器
class SetTransformerEncoder(nn.Module):
    """
    Set Transformer编码器
    使用多层ISAB处理集合输入
    SetTransformerEncoder(X) = ISAB(ISAB(...ISAB(X)))
    """
    def __init__(
        self,
        dim_input: int = 32, # 输入维度 （默认32）
        dim_hidden: int = 32, # 隐藏层维度
        num_heads: int = 4, # 注意力头数 （默认4）
        num_inds: int = 16, # 诱导点数量 （默认16）
        num_layers: int = 3, # ISAB层数 （默认3）
        ln: bool = True, # 是否使用层归一化 （默认True）
    ):
        super(SetTransformerEncoder, self).__init__()
        
        # 第一层：从输入维度映射到隐藏维度
        enc_layers = [ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln)] # 第一层ISAB
        
        # 中间层：保持隐藏维度
        for _ in range(1, num_layers):
            enc_layers.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)) # 中间层ISAB
        
        self.enc = nn.Sequential(*enc_layers) # 编码器
    
    def forward(self, X): # 前向传播    
        """
        Args:
            X: (batch_size, N, dim_input) 细胞嵌入集合
        Returns:
            (batch_size, N, dim_hidden) 编码后的细胞表示
        """
        return self.enc(X) # 编码器 （X）


# 完整的Set Transformer模型
