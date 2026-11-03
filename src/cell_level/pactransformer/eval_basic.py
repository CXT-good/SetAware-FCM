"""
阶段二模型评估脚本：
- 对测试集中的每个样本管做推理
- 为每个样本管生成预测结果
- 基于所有细胞统计每个类别的 accuracy / precision / recall / F1
"""
from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.flow_dataset import TubeDataset
from cell_level.pactransformer.model import SetTransformer
from dataio.marker_mapping import CLASS_LIST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估阶段二 Set Transformer 模型")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的阶段二模型 checkpoint (.pt)")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="嵌入向量目录")
    parser.add_argument("--labels_dir", type=str, default=None, help="标签目录（可选）")
    parser.add_argument("--xuhao_dir", type=str, default=None, help="序号目录（可选）")
    parser.add_argument("--output_dir", type=str, default="./outputs/stage2_eval", help="评估输出目录")
    parser.add_argument("--batch_size", type=int, default=1, help="推理 batch size")
    parser.add_argument("--device", type=int, default=None, help="GPU 编号；不传则自动选择 CUDA/CPU")
    parser.add_argument("--num_classes", type=int, default=len(CLASS_LIST), help="覆盖 checkpoint 中的 num_classes 配置")
    parser.add_argument("--timing_warmup_batches", type=int, default=5, help="不计入统计的 warmup batch 数（纯前向计时）")
    parser.add_argument("--timing_output", type=str, default="inference_timing.json", help="推理计时 JSON 文件名（写到 output_dir 下）")
    return parser.parse_args()


def load_model(model_path: str, num_classes: int, device: torch.device) -> SetTransformer:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = SetTransformer(
        dim_input=checkpoint.get("dim_input", 32),
        dim_hidden=checkpoint.get("dim_hidden", 32),
        num_heads=checkpoint.get("num_heads", 4),
        num_inds=checkpoint.get("num_inds", 16),
        num_layers=checkpoint.get("num_layers", 3),
        num_classes=num_classes,
        dropout=checkpoint.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_inference(
    model: SetTransformer,
    dataloader: DataLoader,
    dataset: TubeDataset,
    device: torch.device,
    num_classes: int,
    output_dir: str,
    timing_warmup_batches: int = 5,
    timing_output_name: str = "inference_timing.json",
) -> np.ndarray:
    """运行推理并返回混淆矩阵"""
    predictions_dir = os.path.join(output_dir, "per_tube_predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    labels = list(range(num_classes))
    agg_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    timing = {
        "task": "inference_forward_only",
        "model": "SetTransformer",
        "device": str(device),
        "batch_size": int(getattr(dataloader, "batch_size", -1) or -1),
        "warmup_batches": int(timing_warmup_batches),
        "timed_batches": 0,
        "timed_samples": 0,
        "forward_seconds_total": 0.0,
    }
    seen_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Stage2 推理")):
            embeddings = batch["embeddings"].to(device)  # (1, N, 32)
            labels_batch = batch["labels"].to(device)  # (1, N)
            xuhao = batch["xuhao"]  # (1, N)
            tube_id = dataset.tube_ids[batch_idx]  # 从dataset获取tube_id

            # 前向传播
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(embeddings)  # (1, N, num_classes)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            seen_batches += 1
            if seen_batches > timing_warmup_batches:
                timing["timed_batches"] += 1
                # timed_samples: number of cells in this tube batch
                timing["timed_samples"] += int(embeddings.shape[1])
                timing["forward_seconds_total"] += float(t1 - t0)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (1, N, num_classes)
            preds = probs.argmax(axis=-1).flatten()  # (N,)
            true_labels = labels_batch.cpu().numpy().flatten()  # (N,)

            # 计算混淆矩阵
            tube_cm = confusion_matrix(
                true_labels,
                preds,
                labels=labels,
            )
            agg_confusion[:] = agg_confusion + tube_cm

            # 保存每个样本管的预测结果（包含概率分布）
            prob_distributions = [json.dumps(prob_list, ensure_ascii=False) for prob_list in probs[0].tolist()]
            tube_results = pd.DataFrame({
                "xuhao": xuhao.flatten(),
                "true_label": true_labels,
                "pred_label": preds,
                "probabilities": prob_distributions,
            })
            tube_results_path = os.path.join(predictions_dir, f"{tube_id}_predict.csv")
            tube_results.to_csv(tube_results_path, index=False)

    if not agg_confusion.any():
        raise RuntimeError("未生成任何预测结果，请确认测试集目录中是否包含数据。")
    
    # 保存混淆矩阵
    np.save(os.path.join(output_dir, "agg_confusion.npy"), agg_confusion)

    # 写出推理计时
    try:
        timing["forward_seconds_per_sample"] = (
            timing["forward_seconds_total"] / timing["timed_samples"]
            if timing["timed_samples"] > 0
            else None
        )
        timing_path = os.path.join(output_dir, timing_output_name)
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2, ensure_ascii=False)
        print(f"推理计时已保存至: {timing_path}")
    except Exception as e:
        print(f"警告：写入推理计时失败: {e}")

    return agg_confusion


def compute_per_class_metrics_from_confusion(
    cm: np.ndarray,
    output_dir: str,
) -> None:
    """从混淆矩阵计算每类指标"""
    num_classes = cm.shape[0]
    labels = list(range(num_classes))
    total = cm.sum()

    tp = np.diag(cm).astype(np.float64)
    row_sums = cm.sum(axis=1).astype(np.float64)
    col_sums = cm.sum(axis=0).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(tp, col_sums, out=np.zeros_like(tp), where=col_sums != 0)
        recall = np.divide(tp, row_sums, out=np.zeros_like(tp), where=row_sums != 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(tp),
            where=(precision + recall) != 0,
        )

    accuracies = []
    for i in range(num_classes):
        fn = row_sums[i] - tp[i]
        fp = col_sums[i] - tp[i]
        tn = total - (tp[i] + fn + fp)
        accuracies.append((tp[i] + tn) / total if total > 0 else 0.0)

    metrics_df = pd.DataFrame(
        {
            "label": labels,
            "accuracy": accuracies,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": row_sums.astype(int),
        }
    )
    metrics_df.loc["overall"] = {
        "label": "overall",
        "accuracy": float(tp.sum() / total) if total > 0 else 0.0,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "support": int(total),
    }
    metrics_path = os.path.join(output_dir, "metrics_per_class.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"每类指标已保存至: {metrics_path}")


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.device}" if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"使用设备: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    num_classes = args.num_classes
    print(f"评估类别数: {num_classes}")

    model = load_model(args.model_path, num_classes=num_classes, device=device)
    
    # 加载数据
    print("加载数据...")
    dataset = TubeDataset(
        embeddings_dir=args.embeddings_dir,
        labels_dir=args.labels_dir,
        xuhao_dir=args.xuhao_dir
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    print(f"测试样本管数量: {len(dataset)}")

    # 运行推理
    agg_confusion = run_inference(
        model,
        dataloader,
        dataset,
        device=device,
        num_classes=num_classes,
        output_dir=args.output_dir,
        timing_warmup_batches=args.timing_warmup_batches,
        timing_output_name=args.timing_output,
    )

    # 计算每类指标
    compute_per_class_metrics_from_confusion(
        agg_confusion, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
