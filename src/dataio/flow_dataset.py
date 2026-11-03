"""
通用流式细胞术数据加载器
支持不同标记物组合和标签列
确保每个batch的数据来自同一个CSV文件
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, BatchSampler
import os
from typing import List, Iterator, Optional, Dict
import random
from collections import defaultdict, deque

from dataio.marker_mapping import MARKER_TO_INDEX, CLASS_TO_INDEX


def _read_csv_with_fallback(data_path: str, **kwargs) -> pd.DataFrame:
    """尝试多种编码读取 CSV，兼容 GBK 等非 UTF-8 文件"""
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"):
        try:
            return pd.read_csv(data_path, encoding=encoding, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "无法用常见编码解析")


class FlowCytometryDataset(Dataset):
    """
    流式细胞术数据集
    直接使用指定的标记物列和标签列，处理单个CSV文件
    """
    
    def __init__(
        self,
        data_path: str,
        marker_columns: Optional[List[str]],
        label_column: str,
        class_to_index: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            data_path: 单个CSV文件的路径
            marker_columns: 标记物列名称列表（CSV文件必须包含这些列）
            label_column: 标签列名称（CSV文件必须包含此列）
        """
        self.data_path = data_path
        env_marker_columns = os.environ.get("FCM_ML_MARKER_COLUMNS", "").strip()
        if env_marker_columns:
            marker_columns = [c.strip() for c in env_marker_columns.split(",") if c.strip()]
            print(f"?????? FCM_ML_MARKER_COLUMNS ?? marker ?: {marker_columns}")
        self.marker_columns = marker_columns
        self.label_column = label_column
        self.class_to_index = class_to_index if class_to_index is not None else CLASS_TO_INDEX

        # 检查是否为单个CSV文件
        if not os.path.isfile(data_path):
            raise ValueError(f"data_path 必须是一个CSV文件: {data_path}")
        if not data_path.endswith('.csv'):
            raise ValueError(f"data_path 必须是CSV文件: {data_path}")
        
        print(f"正在加载CSV文件: {os.path.basename(data_path)}")
        
        # 只读取表头检查列是否存在（支持 GBK 等编码）
        df_header = _read_csv_with_fallback(data_path, nrows=0)
            
        # 检查必需的列是否存在
        header_columns = list(df_header.columns)
        # 自动列规则：除 label / Time / event_ID 外，其余列都作为特征列读取
        ignore_cols = {label_column, "Time", "event_ID"}

        if label_column not in header_columns:
            raise ValueError(f"缺少标签列: {label_column}")

        # 确定可用的标记物列
        if marker_columns is None:
            # 读取所有列（按 ignore_cols 排除），但只保留能映射到 MARKER_TO_INDEX 的列
            candidates = [c for c in header_columns if c not in ignore_cols]
            unknown = [c for c in candidates if c not in MARKER_TO_INDEX]
            if unknown:
                print(f"  警告: 以下列不在 MARKER_TO_INDEX 中，将跳过: {unknown}")
            self.file_marker_columns = [c for c in candidates if c in MARKER_TO_INDEX]
        else:
            missing_markers = [col for col in marker_columns if col not in header_columns]
            if missing_markers:
                print(f"  警告: 缺少标记物列: {missing_markers}")
            # 只取 marker_columns 与表头交集，再按 ignore_cols 排除
            candidates = [col for col in header_columns if col in marker_columns and col not in ignore_cols]
            self.file_marker_columns = candidates
            
        # 读取文件获取行数（只读取必要列）；label 强制按字符串读入，与 class_to_index 键一致
        available_columns = list(self.file_marker_columns) + [label_column]
        df = _read_csv_with_fallback(
            data_path,
            usecols=available_columns,
            dtype={label_column: str},
        )
        df[label_column] = df[label_column].astype(str).str.strip()
        for col in self.file_marker_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        self.num_rows = len(df)

        if self.class_to_index:
            known = set(self.class_to_index)
            labels_in_file = set(df[label_column].unique()) - {"", "nan"}
            unknown = sorted(l for l in labels_in_file if l not in known)
            if unknown:
                n_bad = int(df[label_column].isin(unknown).sum())
                print(
                    f"  警告: {os.path.basename(data_path)} 有 {n_bad} 个细胞 label 不在 class_mapping 中"
                    f"（示例: {unknown[:5]}），将映射为类别 0"
                )

        # 缓存整个DataFrame（因为会多次访问）
        self._df = df
            
        print(f"  - 文件: {os.path.basename(data_path)}: {self.num_rows} 行")
            
        
        self.marker_to_idx = MARKER_TO_INDEX
        self.num_markers = len(self.file_marker_columns)
        
        print(f"\n数据加载完成: {self.num_rows} 个细胞")
        print(f"标记物数量: {self.num_markers}")
        print(f"标记物列表: {self.file_marker_columns[:10]}..." if len(self.file_marker_columns) > 10 else f"标记物列表: {self.file_marker_columns}")
        print(f"标签列: {label_column}")
    
    def __len__(self):
        return self.num_rows
    
    def __getitem__(self, idx):
        """
        返回单个细胞的数据
        从CSV文件中读取数据
        返回: (marker_indices, marker_values, label)
        """
        if idx >= self.num_rows:
            raise IndexError(f"索引 {idx} 超出范围 (总行数: {self.num_rows})")
        
        # 直接从缓存的DataFrame中获取行数据
        row_data = self._df.iloc[idx]
        
        # 提取标记物数据
        marker_data = row_data[self.file_marker_columns].values.astype(np.float32)
        
        # 处理NaN（填充为0）
        marker_data = np.nan_to_num(marker_data, nan=0.0)
        
        marker_indices = torch.LongTensor([
            self.marker_to_idx[col] for col in self.file_marker_columns
        ])
        marker_values = torch.FloatTensor(marker_data)
        
        # 提取标签（已在读 CSV 时转为 str）
        label_key = row_data[self.label_column]
        if not label_key or label_key.lower() == "nan":
            label_idx = 0
        else:
            label_idx = self.class_to_index.get(label_key, 0)
        label = torch.LongTensor([label_idx])

        return {
            'marker_indices': marker_indices,
            'marker_values': marker_values,
            'label': label,
            'num_markers': len(marker_indices),
            'file_path': self.data_path,
            'file_row_idx': idx,
        }



class TubeDataset(Dataset):
    """
    按样本管组织的数据集
    用于阶段二的Set Transformer训练
    """
    
    def __init__(
        self,
        embeddings_dir: str,
        labels_dir: Optional[str] = None,
        xuhao_dir: Optional[str] = None,
        tube_ids: Optional[List[str]] = None
    ):
        """
        Args:
            embeddings_dir: 存储嵌入向量的目录（.npy文件）
            labels_dir: 存储标签的目录（.npy文件），如果为None则从文件名推断
            xuhao_dir: 存储序号的目录（.npy文件），如果为None则从文件名推断
            tube_ids: 要加载的样本管ID列表，如果为None则加载所有
        """
        self.embeddings_dir = embeddings_dir
        self.labels_dir = labels_dir
        self.xuhao_dir = xuhao_dir  
        # 获取所有嵌入文件
        if tube_ids is None:
            self.tube_ids = [
                f.replace('.npy', '') for f in os.listdir(embeddings_dir)
                if f.endswith('.npy')
            ]
        else:
            self.tube_ids = tube_ids
        
        print(f"加载 {len(self.tube_ids)} 个样本管")
    
    def __len__(self):
        return len(self.tube_ids)
    
    def __getitem__(self, idx):
        tube_id = self.tube_ids[idx]
        
        # 加载嵌入向量
        embeddings_path = os.path.join(self.embeddings_dir, f"{tube_id}.npy")
        embeddings = np.load(embeddings_path)  # Shape: (N, 32)
        
        # 加载标签
        if self.labels_dir:
            # 如果 tube_id 以 _embeddings 结尾，替换为 _labels
            if tube_id.endswith('_embeddings'):
                label_tube_id = tube_id.replace('_embeddings', '_labels')
            else:
                label_tube_id = tube_id
            labels_path = os.path.join(self.labels_dir, f"{label_tube_id}.npy")
            labels = np.load(labels_path)
        else:
            # 尝试从嵌入文件名推断标签文件
            labels_path = embeddings_path.replace('embeddings', 'labels')
            if os.path.exists(labels_path):
                labels = np.load(labels_path)
            else:
                labels = np.zeros(len(embeddings), dtype=np.int64)

        # 加载序号
        if self.xuhao_dir:
            # 如果 tube_id 以 _embeddings 结尾，替换为 _xuhao
            if tube_id.endswith('_embeddings'):
                xuhao_tube_id = tube_id.replace('_embeddings', '_xuhao')
            else:
                xuhao_tube_id = tube_id
            xuhao_path = os.path.join(self.xuhao_dir, f"{xuhao_tube_id}.npy")
            xuhao = np.load(xuhao_path)
        else:
            # 尝试从嵌入文件名推断序号文件
            xuhao_path = embeddings_path.replace('embeddings', 'xuhao')
            if os.path.exists(xuhao_path):
                xuhao = np.load(xuhao_path)
            else:
                xuhao = np.zeros(len(embeddings), dtype=np.int64)
       
        # 处理 xuhao：如果是字符串数组，先转换为浮点数再转换为整数
        if isinstance(xuhao, np.ndarray):
            if xuhao.dtype.kind == 'U':  # 字符串类型
                xuhao = xuhao.astype(float).astype(np.int64)
            elif xuhao.dtype.kind != 'i':  # 如果不是整数类型
                xuhao = xuhao.astype(np.int64)
        else:
            xuhao = np.array(xuhao, dtype=np.int64)
        
        return {
            'embeddings': torch.FloatTensor(embeddings),
            'labels': torch.LongTensor(labels),
            'xuhao': torch.LongTensor(xuhao)
        }



