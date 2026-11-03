"""
阶段二训练脚本：基于Set Transformer的分布建模分类
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

from dataio.flow_dataset import TubeDataset
from cell_level.pactransformer.model import SetTransformer
from dataio.marker_mapping import CLASS_LIST, CLASS_NUMBER


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0.0
    epoch_f1 = 0.0
    epoch_precision = 0.0
    epoch_recall = 0.0
    total_samples = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training"), start=1):
        embeddings = batch['embeddings'].to(device)  # (1, N, 32)
        labels = batch['labels'].to(device)  # (1, N)
        xuhao = batch['xuhao'].to(device)  # (1, N)
        
        # 前向传播
        logits = model(embeddings)  # (1, N, num_classes)
        
        # 计算损失（统一使用CrossEntropyLoss）
        logits_flat = logits.view(-1, logits.size(-1))  # (N, num_classes)
        labels_flat = labels.view(-1).long()  # (N,)
        loss = criterion(logits_flat, labels_flat)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 预测并计算当前批次指标
        preds = torch.argmax(logits, dim=-1).cpu().numpy()  # (1, N)
        batch_labels = labels.squeeze().cpu().numpy()  # (N,) 或 (1, N)
        # 确保 preds 和 batch_labels 形状一致
        if preds.ndim > 1:
            preds = preds.flatten()
        if batch_labels.ndim > 1:
            batch_labels = batch_labels.flatten()
        f1 = f1_score(batch_labels, preds, average='weighted', zero_division=0)
        precision = precision_score(batch_labels, preds, average='weighted', zero_division=0)
        recall = recall_score(batch_labels, preds, average='weighted', zero_division=0)
        
        batch_size = labels_flat.shape[0]
        epoch_loss += loss.item() * batch_size
        epoch_f1 += f1 * batch_size
        epoch_precision += precision * batch_size
        epoch_recall += recall * batch_size
        total_samples += batch_size
        
        print(
            f"Batch {batch_idx}: "
            f"avg_loss={loss.item():.4f}, "
            f"f1={f1:.4f}, "
            f"precision={precision:.4f}, "
            f"recall={recall:.4f}"
        )
    
    avg_loss = epoch_loss / total_samples if total_samples > 0 else 0.0
    avg_f1 = epoch_f1 / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, avg_f1


def main():
    parser = argparse.ArgumentParser(description='训练阶段二Set Transformer模型')
    parser.add_argument('--embeddings_dir', type=str, required=True, help='嵌入向量目录')
    parser.add_argument('--labels_dir', type=str, default=None, help='标签目录（可选）')
    parser.add_argument('--xuhao_dir', type=str, default=None, help='序号目录（可选）')
    parser.add_argument('--output_dir', type=str, default='./outputs/stage2', help='输出目录')
    parser.add_argument('--dim_input', type=int, default=32, help='输入维度（嵌入维度）')
    parser.add_argument('--dim_hidden', type=int, default=32, help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--num_inds', type=int, default=16, help='诱导点数量')
    parser.add_argument('--num_layers', type=int, default=3, help='ISAB层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小（建议为1）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--num_classes', type=int, default=len(CLASS_LIST), help='分类类别数（会自动检测，如果提供则使用）')
    parser.add_argument('--device', type=int, default=None, help='指定 GPU 编号（如 0 或 1），不传则自动选择')
    
    args = parser.parse_args()
    
    # 设备
    if args.device is not None:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.device}')
        else:
            print(f"警告: 指定了GPU {args.device}，但CUDA不可用，将使用CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    dataset = TubeDataset(
        embeddings_dir=args.embeddings_dir,
        labels_dir=args.labels_dir,
        xuhao_dir=args.xuhao_dir
    )
    

    num_classes = args.num_classes
    print(f"使用指定的类别数: {num_classes}")
    
    # 使用整个数据集作为训练集
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # 创建模型
    model = SetTransformer(
        dim_input=args.dim_input,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        num_inds=args.num_inds,
        num_layers=args.num_layers,
        num_classes=num_classes,  # 使用检测到的类别数
        dropout=args.dropout
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"分类类别数: {num_classes}")
    
    # 损失函数和优化器
    # 根据CLASS_NUMBER自动计算类别权重（权重与类别数量成反比）
    if len(CLASS_NUMBER) == num_classes and len(CLASS_NUMBER) > 0:
        # 将numpy类型转换为Python int，并计算总样本数
        class_counts = [int(count) for count in CLASS_NUMBER]
        total_samples = sum(class_counts)
        
        # 计算权重：权重 = 总样本数 / (类别数 * 该类别样本数)
        # 这样样本数越少的类别，权重越大
        class_weights = torch.tensor([
            total_samples / (num_classes * count) if count > 0 else 1.0
            for count in class_counts
        ], dtype=torch.float32, device=device)
        
        # 归一化权重，使平均权重为1.0
        class_weights = class_weights / class_weights.mean()
        
        print(f"类别样本数量: {class_counts}")
        print(f"类别权重: {class_weights.cpu().numpy()}")
    else:
        # 如果CLASS_NUMBER不可用，使用平衡权重
        print(f"警告: CLASS_NUMBER长度({len(CLASS_NUMBER)})与类别数({num_classes})不匹配，使用平衡权重")
        class_weights = torch.ones(num_classes, dtype=torch.float32, device=device)
        print(f"类别权重: {class_weights.cpu().numpy()}")
    
    # 损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    print("\n开始训练...")
    epoch_losses = []
    epoch_f1s = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练
        avg_loss, avg_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Epoch {epoch+1} summary: avg_loss={avg_loss:.4f}, avg_f1={avg_f1:.4f}")
        epoch_losses.append(avg_loss)
        epoch_f1s.append(avg_f1)
        # 实时落盘，便于外部脚本监控 loss 谷点
        np.save(os.path.join(args.output_dir, "epoch_losses.npy"), epoch_losses)
        np.save(os.path.join(args.output_dir, "epoch_f1s.npy"), epoch_f1s)
        
        # 学习率调度
        scheduler.step()
        
        # 每隔5个epoch保存一次权重
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'num_classes': num_classes,
                'dim_input': args.dim_input,
                'dim_hidden': args.dim_hidden,
                'num_heads': args.num_heads,
                'num_inds': args.num_inds,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
            }, checkpoint_path)
            print(f"已保存权重: {checkpoint_path}")

    # 兜底保存 epoch 曲线
    np.save(os.path.join(args.output_dir, "epoch_losses.npy"), epoch_losses)
    np.save(os.path.join(args.output_dir, "epoch_f1s.npy"), epoch_f1s)
    print(f"epoch_losses和epoch_f1s已保存至: {args.output_dir}")

    if epoch_losses:
        epochs_range = range(1, args.epochs + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, epoch_losses, label="avg_loss", marker="o")
        plt.plot(epochs_range, epoch_f1s, label="avg_f1", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Metric value")
        plt.title("Training Metrics Trend")
        plt.legend()
        plt.grid(alpha=0.3)
        plot_path = os.path.join(args.output_dir, "training_metrics.pdf")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"训练曲线已保存至: {plot_path}")

    print(f"\n训练完成！")


if __name__ == '__main__':
    main()

