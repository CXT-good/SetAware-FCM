"""
通用流式细胞术数据加载器
支持不同标记物组合和标签列
确保每个batch的数据来自同一个CSV文件
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, Sampler, BatchSampler, get_worker_info
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


def _marker_values_to_float32(df: pd.DataFrame) -> np.ndarray:
    """将 marker 列转为 float32；非法字符串（如 '10.1281p202s115634'）置为 NaN 再填 0。"""
    numeric = df.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float32, copy=False)
    return np.nan_to_num(values, nan=0.0)

class UnlabeledFlowCytometryDataset(Dataset):
    """
    无标签的流式细胞术数据集
    直接使用指定的标记物列，处理单个CSV文件
    """
    
    def __init__(
        self,
        data_path: str,
        marker_columns: List[str],
        *,
        verbose: bool = True,
        max_rows: int = 0,
    ):
        """
        Args:
            data_path: 单个CSV文件的路径
            marker_columns: 标记物列名称列表（CSV文件必须包含这些列）
            max_rows: 单文件最多读取多少行；<=0 表示不限。
                对于超大 CSV（如 100w 行级），用它限制单次读入内存的行数，
                避免一个 worker 加载就吃 2.6 GB（100w * 649 * float32），
                导致 64 个 worker 总内存爆炸 / GPU 显存吃紧。
        """
        self.data_path = data_path
        self.marker_columns = marker_columns
        self.verbose = verbose
        self.max_rows = int(max_rows) if max_rows else 0

        # 检查是否为单个CSV文件
        if not os.path.isfile(data_path):
            raise ValueError(f"data_path 必须是一个CSV文件: {data_path}")
        if not data_path.endswith('.csv'):
            raise ValueError(f"data_path 必须是CSV文件: {data_path}")
        
        if verbose:
            print(f"正在加载CSV文件: {os.path.basename(data_path)}")
        
        # 只读取表头检查列是否存在（支持 GBK 等编码）
        df_header = _read_csv_with_fallback(data_path, nrows=0)
            
        # 检查必需的列是否存在
        header_columns = list(df_header.columns)
        missing_markers = [col for col in marker_columns if col not in header_columns]
        if missing_markers and verbose:
            if len(missing_markers) <= 8:
                print(f"  警告: 缺少标记物列: {missing_markers}")
            else:
                print(f"  警告: 缺少 {len(missing_markers)} 个标记物列（仅显示前 5 个）: {missing_markers[:5]}")
            
        # 确定可用的标记物列
        self.file_marker_columns = [col for col in header_columns if col in marker_columns]
            
        # 读取文件获取行数（只读取必要列）
        read_kwargs = {"usecols": self.file_marker_columns}
        if self.max_rows > 0:
            read_kwargs["nrows"] = self.max_rows
        df = _read_csv_with_fallback(data_path, **read_kwargs)
        self.num_rows = len(df)

        values = _marker_values_to_float32(df)
        self._values = values
        del df

        self.marker_to_idx = MARKER_TO_INDEX
        self.num_markers = len(self.file_marker_columns)
        self._marker_indices = torch.LongTensor(
            [self.marker_to_idx[col] for col in self.file_marker_columns]
        )

        if verbose:
            print(f"  - 文件: {os.path.basename(data_path)}: {self.num_rows} 行")
            print(f"\n数据加载完成: {self.num_rows} 个细胞")
            print(f"标记物数量: {self.num_markers}")
            preview = (
                f"{self.file_marker_columns[:10]}..."
                if len(self.file_marker_columns) > 10
                else f"{self.file_marker_columns}"
            )
            print(f"标记物列表: {preview}")
    
    def __len__(self):
        return self.num_rows
    
    def __getitem__(self, idx):
        """
        返回单个细胞的数据
        从CSV文件中读取数据
        返回: (marker_indices, marker_values)
        """
        if idx >= self.num_rows:
            raise IndexError(f"索引 {idx} 超出范围 (总行数: {self.num_rows})")

        return {
            'marker_indices': self._marker_indices,
            'marker_values': torch.from_numpy(self._values[idx]),
            'num_markers': self.num_markers,
            'file_path': self.data_path,
            'file_row_idx': idx,
        }


class PretrainEpochBatchIterableDataset(IterableDataset):
    """
    按 epoch 遍历多个 CSV，每个 batch 仍来自同一文件。
    配合 persistent_workers 使用：整个 epoch 只创建一次 DataLoader。
    """

    def __init__(
        self,
        csv_files: List[str],
        file_order: List[int],
        marker_columns: List[str],
        batch_size: int,
        rank: int,
        world_size: int,
        epoch: int,
        seed: int,
        *,
        verbose: bool = False,
        max_rows_per_file: int = 0,
    ) -> None:
        self.csv_files = csv_files
        self.file_order = file_order
        self.marker_columns = marker_columns
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.epoch = epoch
        self.seed = seed
        self.verbose = verbose
        self.max_rows_per_file = int(max_rows_per_file) if max_rows_per_file else 0

    def __iter__(self) -> Iterator[dict]:
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        rng = random.Random(
            self.seed + self.epoch * 1_000_003 + self.rank * 10_007 + worker_id
        )

        for ord_i, file_idx in enumerate(self.file_order):
            if ord_i % num_workers != worker_id:
                continue

            try:
                ds = UnlabeledFlowCytometryDataset(
                    self.csv_files[file_idx],
                    self.marker_columns,
                    verbose=self.verbose and worker_id == 0,
                    max_rows=self.max_rows_per_file,
                )
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                if self.verbose and worker_id == 0:
                    print(
                        f"  跳过无法解析的文件: {os.path.basename(self.csv_files[file_idx])} ({exc})",
                        flush=True,
                    )
                continue
            rank_indices = [i for i in range(len(ds)) if i % self.world_size == self.rank]
            rng.shuffle(rank_indices)

            batch_indices: list[int] = []
            for row_idx in rank_indices:
                batch_indices.append(row_idx)
                if len(batch_indices) == self.batch_size:
                    yield self._collate_batch(ds, batch_indices, file_idx)
                    batch_indices = []
            if batch_indices:
                yield self._collate_batch(ds, batch_indices, file_idx)

    @staticmethod
    def _collate_batch(ds: UnlabeledFlowCytometryDataset, indices: List[int], file_idx: int) -> dict:
        samples = [ds[i] for i in indices]
        return {
            "marker_indices": torch.stack([s["marker_indices"] for s in samples]),
            "marker_values": torch.stack([s["marker_values"] for s in samples]),
            "file_index": file_idx,
        }


class LabeledFlowCytometryDataset(Dataset):
    """
    带标签的流式细胞术数据集
    直接使用指定的标记物列和标签列，处理单个CSV文件
    """
    
    def __init__(
        self,
        data_path: str,
        marker_columns: List[str],
        label_column: str,
        xuhao_column: str,
    ):
        """
        Args:
            data_path: 单个CSV文件的路径
            marker_columns: 标记物列名称列表（CSV文件必须包含这些列）
            label_column: 标签列名称（CSV文件必须包含此列）
            xuhao_column: 序号列名称（CSV文件必须包含此列）
        """
        self.data_path = data_path
        self.marker_columns = marker_columns
        self.label_column = label_column
        self.xuhao_column = xuhao_column

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
        missing_markers = [col for col in marker_columns if col not in header_columns]
        if missing_markers:
            print(f"  警告: 缺少标记物列: {missing_markers}")
            
        if label_column not in header_columns:
            raise ValueError(f"缺少标签列: {label_column}")
            
        # 序号列是可选的，如果为空字符串则跳过检查
        if xuhao_column and xuhao_column.strip() != "":
            if xuhao_column not in header_columns:
                raise ValueError(f"缺少序号列: {xuhao_column}")
            
        # 确定可用的标记物列
        self.file_marker_columns = [col for col in header_columns if col in marker_columns]
            
        # 读取文件获取行数（只读取必要列）
        available_columns = self.file_marker_columns + [label_column]
        if xuhao_column and xuhao_column.strip() != "":
            available_columns.append(xuhao_column)
        df = _read_csv_with_fallback(data_path, usecols=available_columns)
        self.num_rows = len(df)
            
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
        
        # 提取标签
        label_val = row_data[self.label_column]
        if pd.isna(label_val):
            label_val = 0
        else:
            # 如果标签是字符串（中文类别名称），需要映射到索引
            if isinstance(label_val, str):
                label_val = CLASS_TO_INDEX.get(label_val, 0)
            else:
                label_val = int(label_val)
        label = torch.LongTensor([label_val])
        
        # 提取序号（如果序号列存在）
        if self.xuhao_column and self.xuhao_column.strip() != "":
            xuhao = str(row_data[self.xuhao_column])
        else:
            xuhao = str(idx)  # 如果没有序号列，使用行索引作为序号

        return {
            'marker_indices': marker_indices,
            'marker_values': marker_values,
            'label': label,
            'num_markers': len(marker_indices),
            'file_path': self.data_path,
            'file_row_idx': idx,
            'xuhao': xuhao
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



