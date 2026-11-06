"""
从阶段一模型提取嵌入向量，用于阶段二训练
按样本管组织数据
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import json
from typing import Optional

from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]

from dataio.flow_dataset import FlowCytometryDataset
from cell_level.ctransformer.model import Stage1Transformer
from dataio.marker_mapping import MARKER_INDEX_LIST, MARKER_LIST, LABEL_COLUMN, default_class_mapping_path


def _default_class_mapping_path() -> str:
    return default_class_mapping_path()


def _class_mapping_from_checkpoint(checkpoint: dict, model_path: str) -> Optional[dict[str, int]]:
    class_to_index = checkpoint.get("class_to_index")
    if isinstance(class_to_index, dict) and class_to_index:
        return {str(k): int(v) for k, v in class_to_index.items()}

    class_list = checkpoint.get("class_list")
    if isinstance(class_list, (list, tuple)) and class_list:
        return {str(name): idx for idx, name in enumerate(class_list)}

    dataset_key = checkpoint.get("dataset_key")
    mapping_path = _default_class_mapping_path()
    if dataset_key and os.path.isfile(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mappings = json.load(f)
        spec = mappings.get("datasets", {}).get(str(dataset_key), {})
        class_list = spec.get("class_list")
        if isinstance(class_list, list) and class_list:
            return {str(name): idx for idx, name in enumerate(class_list)}

    print(f"警告: {model_path} 中没有可用 class_to_index/class_list/dataset_key，提取时将使用数据集默认映射。")
    return None


def extract_embeddings_by_tube(
    model_path: str,
    data_dir: str,
    output_dir: str,
    batch_size: int = 32,
    device_id: Optional[int] = None,
    num_workers: int = 4,
):
    """
    提取嵌入向量并按样本管组织
    
    Args:
        model_path: 阶段一模型路径
        data_dir: 数据路径（CSV文件或目录）
        output_dir: 输出目录
        batch_size: 批次大小
        device_id: 设备ID
        num_workers: 数据加载器使用的进程数
    """
    if device_id is not None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{device_id}")
        else:
            print("警告: 指定了 GPU 但当前环境无可用 CUDA，改用 CPU。")
            device = torch.device("cpu")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_markers = checkpoint['num_markers']
    # 从checkpoint中读取类别数（如果存在），否则使用默认值2
    num_classes = checkpoint.get('num_classes', 2)
    class_to_index = _class_mapping_from_checkpoint(checkpoint, model_path)
    print(f"  标记物数量: {num_markers}")
    print(f"  类别数量: {num_classes}")
    if class_to_index:
        print(f"  类别映射: {class_to_index}")
    
    model = Stage1Transformer(
        markers=MARKER_INDEX_LIST,
        embed_dim=checkpoint.get("embed_dim", 32),
        num_heads=checkpoint.get("num_heads", 4),
        num_layers=checkpoint.get("num_layers", 3),
        dim_feedforward=checkpoint.get("dim_feedforward", 128),
        dropout=checkpoint.get("dropout", 0.1),
        num_classes=num_classes,
        value_embedding_mode=checkpoint.get("value_embedding_mode", "shared"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    embeddings_dir = os.path.join(output_dir, 'embeddings')
    labels_dir = os.path.join(output_dir, 'labels')
    xuhao_dir = os.path.join(output_dir, 'xuhao')
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(xuhao_dir, exist_ok=True)
    # 加载数据
    print("加载数据...")
    if os.path.isdir(data_dir):
        data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        base_path = data_dir
    else:
        data_files = [os.path.basename(data_dir)]
        base_path = os.path.dirname(data_dir) if os.path.dirname(data_dir) else '.'
    
    # 处理每个文件
    for file in data_files:
        file_path = os.path.join(base_path, file) if os.path.isdir(base_path) else data_dir
        print(f"\n处理文件: {file}")
        
        # 加载数据
        dataset = FlowCytometryDataset(
            data_path=file_path,
            marker_columns=MARKER_LIST,
            label_column=LABEL_COLUMN,
            class_to_index=class_to_index,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        # 提取嵌入向量
        all_embeddings = []
        all_labels = []
        all_xuhao = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取嵌入"):
                marker_indices = batch['marker_indices'][0].to(device)
                marker_values = batch['marker_values'].to(device)
                labels = batch['label']
                row_idx = batch['file_row_idx']
                # 提取嵌入
                _, embeddings = model(marker_values, marker_indices)
                embeddings = embeddings.cpu().numpy()
                
                all_embeddings.append(embeddings)
                all_labels.append(labels.numpy())
                all_xuhao.extend([int(x) for x in row_idx])
        # 合并
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0).flatten()
        all_xuhao = np.array(all_xuhao)
        # 整个文件是一个样本管
        tube_name = os.path.splitext(file)[0]
        np.save(os.path.join(embeddings_dir, f"{tube_name}_embeddings.npy"), all_embeddings)
        np.save(os.path.join(labels_dir, f"{tube_name}_labels.npy"), all_labels)
        np.save(os.path.join(xuhao_dir, f"{tube_name}_xuhao.npy"), all_xuhao)
        print(f"  保存样本管 {tube_name}: {len(all_embeddings)} 个细胞")
    
    print(f"\n嵌入向量提取完成！")
    print(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='从阶段一模型提取嵌入向量')
    parser.add_argument('--model_path', type=str, required=True, help='阶段一模型路径')
    parser.add_argument('--data_dir', type=str, required=True, help='数据路径（CSV文件或目录）')
    parser.add_argument('--output_dir', type=str, default='./outputs/embeddings', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--device', type=int, default=None, help='CUDA 设备号，例如 0/1；不传则自动选择')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器使用的进程数')
    args = parser.parse_args()
    
    extract_embeddings_by_tube(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_id=args.device,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()

