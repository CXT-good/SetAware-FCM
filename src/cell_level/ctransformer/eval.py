"""
阶段一模型评估脚本：
- 从 split xlsx 的 test 子表读取 csv_path（与 train.py 的 train 子表对称）
- 使用 checkpoint 中的 class_to_index / 模型结构配置
- 逐 CSV 推理，输出预测、合并表与混淆矩阵指标
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.class_lists import dataset_key_from_split_xlsx, macro_label_indices
from dataio.flow_dataset import FlowCytometryDataset, _read_csv_with_fallback
from dataio.marker_mapping import (
    LABEL_COLUMN,
    MARKER_INDEX_LIST,
    load_class_list_for_dataset,
    default_class_mapping_path,
)
from cell_level.ctransformer.model import Stage1Transformer

STAGE1_MODEL_NAME = "stage1_transformer"
DEFAULT_OUTPUT_ROOT = "outputs"
INIT_MODES: tuple[str, ...] = ("random", "from_pretrain")

# 与 train.py 一致，用于旧 checkpoint 缺少结构字段时回退
MODEL_SIZE_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {
        "embed_dim": 128,
        "num_layers": 6,
        "num_heads": 8,
        "dim_feedforward": 512,
    },
    "small": {
        "embed_dim": 384,
        "num_layers": 8,
        "num_heads": 16,
        "dim_feedforward": 1536,
    },
    "base": {
        "embed_dim": 512,
        "num_layers": 12,
        "num_heads": 16,
        "dim_feedforward": 2048,
    },
}


def _infer_size_and_init_from_path(model_path: str) -> tuple[str, str]:
    """从 checkpoint 路径推断 model_size / init_mode（兼容旧布局）。"""
    path = Path(model_path).resolve()
    parent = path.parent.name
    grandparent = path.parent.parent.name
    if parent in INIT_MODES and grandparent in MODEL_SIZE_PRESETS:
        return grandparent, parent
    if parent in MODEL_SIZE_PRESETS:
        return parent, "random"
    return "unknown", "random"


def resolve_eval_output_dir(
    *,
    output_root: str,
    test_xlsx: str,
    checkpoint: dict,
    model_path: str,
    model_name: str = "",
    output_dir_override: str = "",
) -> str:
    """
    评估输出：output_root / 数据集 / 模型 / model_size / init_mode
    布局与 checkpoint 一致（仅目录用途为评估结果，不含训练权重）。
    """
    if output_dir_override:
        return os.path.abspath(output_dir_override)
    dataset_key = checkpoint.get("dataset_key") or dataset_key_from_split_xlsx(
        test_xlsx
    )
    model = checkpoint.get("model_name") or model_name or STAGE1_MODEL_NAME
    model_size = checkpoint.get("model_size")
    init_mode = checkpoint.get("init_mode", "random")
    if not model_size:
        model_size, init_mode = _infer_size_and_init_from_path(model_path)
    root = os.path.abspath(output_root)
    return os.path.join(root, dataset_key, model, model_size, init_mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估阶段一 Transformer 模型")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练好的阶段一 checkpoint (.pt)",
    )
    parser.add_argument(
        "--test_xlsx",
        type=str,
        required=True,
        help="测试集 split xlsx 的绝对路径（从 test 子表读取 csv_path）",
    )
    parser.add_argument(
        "--xlsx_csv_col",
        type=str,
        default="csv_path",
        help="xlsx 中 CSV 路径列名（默认 csv_path）",
    )
    parser.add_argument(
        "--class_mapping_path",
        type=str,
        default=default_class_mapping_path(),
        help="逐细胞类别映射 JSON（checkpoint 无 class_to_index 时使用）",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="评估结果根目录；实际输出为 <root>/<数据集>/<模型>/<model_size>/<init_mode>/",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="",
        help="（已弃用，请用 --output_root）",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="模型名称目录（默认从 checkpoint 读取，否则 stage1_transformer）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="覆盖自动输出路径；默认按 output_root/数据集/模型/model_size/init_mode 生成",
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="推理 batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker 数")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="GPU 编号；不传则自动选择 CUDA/CPU",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="覆盖 checkpoint 中的 num_classes（一般无需指定）",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default=LABEL_COLUMN,
        help="真实标签列名",
    )
    parser.add_argument(
        "--timing_warmup_batches",
        type=int,
        default=5,
        help="不计入统计的 warmup batch 数（纯前向计时）",
    )
    parser.add_argument(
        "--timing_output",
        type=str,
        default="inference_timing.json",
        help="推理计时 JSON 文件名（写到 output_dir 下）",
    )
    parser.add_argument(
        "--skip_cell_dumps",
        action="store_true",
        help="不写 per_file_predictions / merged_with_truth 逐细胞明细（仅算混淆矩阵与指标，省磁盘）",
    )
    return parser.parse_args()


def _resolve_class_info(
    checkpoint: dict,
    *,
    test_xlsx: str,
    class_mapping_path: str,
) -> tuple[list[str], dict[str, int], int]:
    """从 checkpoint 或 class_mapping 解析 class_list / class_to_index。"""
    class_list = checkpoint.get("class_list")
    class_to_index = checkpoint.get("class_to_index")

    if class_list and class_to_index:
        class_list = list(class_list)
        class_to_index = dict(class_to_index)
    else:
        ds_key = checkpoint.get("dataset_key") or dataset_key_from_split_xlsx(test_xlsx)
        mapping_path = class_mapping_path
        if not os.path.isabs(mapping_path):
            mapping_path = os.path.abspath(mapping_path)
        class_list, _counts = load_class_list_for_dataset(ds_key, mapping_path=mapping_path)
        class_to_index = {c: i for i, c in enumerate(class_list)}
        print(f"checkpoint 无 class_to_index，已从映射文件加载: dataset_key={ds_key}")

    num_classes = len(class_list)
    return class_list, class_to_index, num_classes


def _model_kwargs_from_checkpoint(checkpoint: dict) -> dict:
    """从 checkpoint（及 model_size 预设）构建 Stage1Transformer 构造参数。"""
    model_size = checkpoint.get("model_size", "tiny")
    preset = MODEL_SIZE_PRESETS.get(model_size, MODEL_SIZE_PRESETS["tiny"])
    return {
        "embed_dim": checkpoint.get("embed_dim", preset["embed_dim"]),
        "num_heads": checkpoint.get("num_heads", preset["num_heads"]),
        "num_layers": checkpoint.get("num_layers", preset["num_layers"]),
        "dim_feedforward": checkpoint.get(
            "dim_feedforward", preset["dim_feedforward"]
        ),
        "dropout": checkpoint.get("dropout", 0.1),
        "value_embedding_mode": checkpoint.get("value_embedding_mode", "shared"),
    }


def load_model(
    model_path: str,
    *,
    num_classes: int,
    device: torch.device,
) -> tuple[Stage1Transformer, dict]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    kw = _model_kwargs_from_checkpoint(checkpoint)
    model = Stage1Transformer(
        markers=MARKER_INDEX_LIST,
        num_classes=num_classes,
        **kw,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _collect_test_csv_paths(
    test_xlsx: str,
    *,
    csv_col: str,
) -> list[str]:
    if not os.path.isabs(test_xlsx):
        raise ValueError(f"--test_xlsx 必须是绝对路径: {test_xlsx}")
    if not os.path.isfile(test_xlsx):
        raise FileNotFoundError(f"--test_xlsx 文件不存在: {test_xlsx}")

    df = pd.read_excel(test_xlsx, sheet_name="test")
    if csv_col not in df.columns:
        raise ValueError(
            f"xlsx test 子表缺少列 {csv_col}，当前列: {list(df.columns)}"
        )

    paths: list[str] = []
    missing = 0
    for raw in df[csv_col].astype(str).tolist():
        p = os.path.abspath(raw.strip())
        if not p.lower().endswith(".csv"):
            continue
        if os.path.isfile(p):
            paths.append(p)
        else:
            missing += 1
    if missing:
        print(f"警告: test 子表中有 {missing} 个 csv_path 不存在（已跳过）")
    if not paths:
        raise ValueError("test 子表中没有可用的 CSV 路径")
    return paths


def _build_dataloader(
    dataset: FlowCytometryDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    use_cuda = device.type == "cuda"
    dl_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **dl_kwargs)


def run_inference_on_file(
    model: Stage1Transformer,
    dataloader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
    index_to_class: dict[int, str],
    output_dir: str,
    timing_warmup_batches: int,
    timing_acc: dict,
    skip_cell_dumps: bool = False,
) -> np.ndarray:
    """对单个 CSV 对应的 DataLoader 推理，返回该文件的混淆矩阵。"""
    predictions_dir = os.path.join(output_dir, "per_file_predictions")
    merged_dir = os.path.join(output_dir, "merged_with_truth")
    if not skip_cell_dumps:
        os.makedirs(predictions_dir, exist_ok=True)
        os.makedirs(merged_dir, exist_ok=True)

    header_cols = [
        "file_row_idx",
        "true_label_idx",
        "pred_label_idx",
        "true_label",
        "pred_label",
        "probabilities",
    ]
    class_indices = np.arange(num_classes, dtype=np.int64)

    current_file_name: str | None = None
    current_source_path: str | None = None
    current_pred_path: str | None = None
    current_file_obj = None
    current_writer = None
    file_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def finalize_current_file():
        nonlocal current_file_name, current_file_obj, current_writer
        nonlocal current_source_path, current_pred_path, file_confusion
        if current_file_name is None:
            return
        if skip_cell_dumps:
            # 明细已在 batch 循环中累计到混淆矩阵，无需读回文件
            current_file_name = None
            current_source_path = None
            current_pred_path = None
            return
        if current_file_obj:
            current_file_obj.close()
            current_file_obj = None
        current_writer = None
        assert current_pred_path and current_source_path

        pred_df = pd.read_csv(current_pred_path, encoding="utf-8")
        source_df = _read_csv_with_fallback(current_source_path)
        source_df = source_df.reset_index(drop=True)
        pred_df = pred_df.reset_index(drop=True)
        merged_df = source_df.copy()
        merged_df["true_label_idx"] = pred_df["true_label_idx"]
        merged_df["pred_label_idx"] = pred_df["pred_label_idx"]
        merged_df["true_label"] = pred_df["true_label"]
        merged_df["pred_label"] = pred_df["pred_label"]
        merged_df["probabilities"] = pred_df["probabilities"]
        merged_path = os.path.join(merged_dir, f"{current_file_name}_merged.csv")
        merged_df.to_csv(merged_path, index=False)

        file_confusion += confusion_matrix(
            pred_df["true_label_idx"].to_numpy(),
            pred_df["pred_label_idx"].to_numpy(),
            labels=class_indices,
        )
        current_file_name = None
        current_source_path = None
        current_pred_path = None

    def start_new_file(file_name: str, file_path: str):
        nonlocal current_file_name, current_source_path, current_pred_path
        nonlocal current_file_obj, current_writer
        current_file_name = file_name
        current_source_path = file_path
        if skip_cell_dumps:
            current_pred_path = None
            current_file_obj = None
            current_writer = None
            return
        current_pred_path = os.path.join(predictions_dir, f"{file_name}_predict.csv")
        current_file_obj = open(current_pred_path, "w", newline="", encoding="utf-8")
        current_writer = csv.writer(current_file_obj)
        current_writer.writerow(header_cols)

    seen_batches = 0
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Stage1 推理"):
            non_blocking = use_cuda
            marker_values = batch["marker_values"].to(device, non_blocking=non_blocking)
            marker_indices = batch["marker_indices"][0].to(device, non_blocking=non_blocking)
            batch_labels = batch["label"].squeeze(-1)

            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits, _ = model(marker_values, marker_indices)
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            seen_batches += 1
            if seen_batches > timing_warmup_batches:
                timing_acc["timed_batches"] += 1
                timing_acc["timed_samples"] += int(marker_values.shape[0])
                timing_acc["forward_seconds_total"] += float(t1 - t0)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            true_labels = batch_labels.cpu().numpy()

            file_paths = batch["file_path"]
            row_indices = batch["file_row_idx"]
            if not file_paths:
                continue

            batch_file_path = file_paths[0]
            batch_file_name = Path(batch_file_path).stem
            if batch_file_name != current_file_name:
                finalize_current_file()
                start_new_file(batch_file_name, batch_file_path)

            if skip_cell_dumps:
                # 直接在内存里累计混淆矩阵，不写逐细胞明细
                file_confusion += confusion_matrix(
                    true_labels.astype(np.int64),
                    preds.astype(np.int64),
                    labels=class_indices,
                )
                continue

            for i in range(len(preds)):
                ti = int(true_labels[i])
                pi = int(preds[i])
                prob_list = [float(probs[i, c]) for c in range(num_classes)]
                current_writer.writerow(
                    [
                        int(row_indices[i]),
                        ti,
                        pi,
                        index_to_class.get(ti, str(ti)),
                        index_to_class.get(pi, str(pi)),
                        json.dumps(prob_list, ensure_ascii=False),
                    ]
                )

    finalize_current_file()
    return file_confusion


def plot_confusion_matrix(
    cm: np.ndarray,
    output_dir: str,
    labels: list[str],
) -> None:
    plt.figure(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.5)))
    cm_percent = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) * 100
    plt.imshow(cm_percent, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="百分比 (%)")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                f"{cm_percent[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if cm_percent[i, j] > 50 else "black",
                fontsize=8,
            )
    plt.title("混淆矩阵 (Confusion Matrix)")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵可视化已保存至: {plot_path}")

    plt.figure(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.5)))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="样本数")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                fontsize=8,
            )
    plt.title("混淆矩阵 - 原始计数")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plot_path_counts = os.path.join(output_dir, "confusion_matrix_counts.png")
    plt.savefig(plot_path_counts, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵（原始数值）已保存至: {plot_path_counts}")


def compute_per_class_metrics_from_confusion(
    cm: np.ndarray,
    output_dir: str,
    *,
    class_list: list[str],
    dataset_key: str | None = None,
) -> None:
    num_classes = cm.shape[0]
    display_labels = (
        class_list if len(class_list) == num_classes else [str(i) for i in range(num_classes)]
    )
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

    y_true = []
    y_pred = []
    for i in range(num_classes):
        for j in range(num_classes):
            n = int(cm[i, j])
            if n:
                y_true.extend([i] * n)
                y_pred.extend([j] * n)

    macro_labels = macro_label_indices(
        display_labels, dataset_key, supports=row_sums.astype(int).tolist()
    )
    macro_f1 = macro_p = macro_r = float("nan")
    if y_true and macro_labels:
        macro_f1 = f1_score(
            y_true, y_pred, average="macro", labels=macro_labels, zero_division=0
        )
        macro_p = precision_score(
            y_true, y_pred, average="macro", labels=macro_labels, zero_division=0
        )
        macro_r = recall_score(
            y_true, y_pred, average="macro", labels=macro_labels, zero_division=0
        )

    metrics_df = pd.DataFrame(
        {
            "class_name": display_labels,
            "class_index": list(range(num_classes)),
            "accuracy": accuracies,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": row_sums.astype(int),
        }
    )
    metrics_df.loc["overall"] = {
        "class_name": "overall",
        "class_index": "",
        "accuracy": float(tp.sum() / total) if total > 0 else 0.0,
        "precision": macro_p,
        "recall": macro_r,
        "f1": macro_f1,
        "support": int(total),
    }
    metrics_path = os.path.join(output_dir, "metrics_per_class.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"每类指标已保存至: {metrics_path}")
    print(
        f"macro: precision={macro_p:.4f}, recall={macro_r:.4f}, f1={macro_f1:.4f}"
    )
    plot_confusion_matrix(cm, output_dir, display_labels)


def main() -> None:
    args = parse_args()

    if args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if not os.path.isabs(args.test_xlsx):
        raise ValueError(f"--test_xlsx 必须是绝对路径: {args.test_xlsx}")
    if not os.path.isfile(args.test_xlsx):
        raise FileNotFoundError(f"--test_xlsx 文件不存在: {args.test_xlsx}")

    if not os.path.isabs(args.model_path):
        args.model_path = os.path.abspath(args.model_path)
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"checkpoint 不存在: {args.model_path}")

    class_mapping_path = args.class_mapping_path
    if not os.path.isabs(class_mapping_path):
        class_mapping_path = os.path.abspath(class_mapping_path)

    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)

    if args.result_root:
        output_root = args.result_root
    else:
        output_root = args.output_root

    args.output_dir = resolve_eval_output_dir(
        output_root=output_root,
        test_xlsx=args.test_xlsx,
        checkpoint=checkpoint,
        model_path=args.model_path,
        model_name=args.model_name,
        output_dir_override=args.output_dir,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    ds_key = checkpoint.get("dataset_key") or dataset_key_from_split_xlsx(args.test_xlsx)
    model_name = checkpoint.get("model_name") or args.model_name or STAGE1_MODEL_NAME
    model_size = checkpoint.get("model_size")
    init_mode = checkpoint.get("init_mode", "random")
    if not model_size:
        model_size, init_mode = _infer_size_and_init_from_path(args.model_path)
    print(
        f"评估输出目录: {args.output_dir}  "
        f"(数据集={ds_key}, 模型={model_name}, size={model_size}, init={init_mode})"
    )
    class_list, class_to_index, num_classes = _resolve_class_info(
        checkpoint,
        test_xlsx=args.test_xlsx,
        class_mapping_path=class_mapping_path,
    )
    if args.num_classes is not None:
        num_classes = int(args.num_classes)
    index_to_class = {i: c for c, i in class_to_index.items()}

    print(f"dataset_key={ds_key}")
    print(f"评估类别数: {num_classes}")
    print(f"class_list: {class_list}")

    model, _ = load_model(args.model_path, num_classes=num_classes, device=device)
    kw = _model_kwargs_from_checkpoint(checkpoint)
    print(
        f"模型结构: embed_dim={kw['embed_dim']}, num_layers={kw['num_layers']}, "
        f"num_heads={kw['num_heads']}, dim_feedforward={kw['dim_feedforward']}"
    )

    test_files = _collect_test_csv_paths(args.test_xlsx, csv_col=args.xlsx_csv_col)
    print(f"测试 CSV 数量: {len(test_files)}")

    timing_acc = {
        "task": "inference_forward_only",
        "model": "Stage1Transformer",
        "device": str(device),
        "batch_size": args.batch_size,
        "warmup_batches": args.timing_warmup_batches,
        "timed_batches": 0,
        "timed_samples": 0,
        "forward_seconds_total": 0.0,
        "files": len(test_files),
    }

    agg_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for data_file in test_files:
        print(f"\n评估文件: {os.path.basename(data_file)}")
        dataset = FlowCytometryDataset(
            data_path=data_file,
            marker_columns=None,
            label_column=args.label_column,
            class_to_index=class_to_index,
        )
        dataloader = _build_dataloader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        file_cm = run_inference_on_file(
            model,
            dataloader,
            device,
            num_classes=num_classes,
            index_to_class=index_to_class,
            output_dir=args.output_dir,
            timing_warmup_batches=args.timing_warmup_batches,
            timing_acc=timing_acc,
            skip_cell_dumps=args.skip_cell_dumps,
        )
        agg_confusion += file_cm

    if not agg_confusion.any():
        raise RuntimeError("未生成任何预测结果，请检查 test 子表与 CSV 标签列。")

    np.save(os.path.join(args.output_dir, "agg_confusion.npy"), agg_confusion)

    timing_acc["forward_seconds_per_sample"] = (
        timing_acc["forward_seconds_total"] / timing_acc["timed_samples"]
        if timing_acc["timed_samples"] > 0
        else None
    )
    timing_path = os.path.join(args.output_dir, args.timing_output)
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_acc, f, indent=2, ensure_ascii=False)
    print(f"推理计时已保存至: {timing_path}")

    compute_per_class_metrics_from_confusion(
        agg_confusion,
        args.output_dir,
        class_list=class_list,
        dataset_key=ds_key,
    )


if __name__ == "__main__":
    main()
