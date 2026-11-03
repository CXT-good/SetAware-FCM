"""在验证集 CSV 上评估微调后的 CellClassifier，结果写入实验目录的 eval/。

- ``agg_confusion.npy``：聚合混淆矩阵
- ``metrics_per_class.csv``：每类 accuracy / precision / recall / F1 / support 及 overall 行
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.mae_dataset import LabeledFlowCytometryDataset
from mae.layout import (
    MODEL_SIZE_NAMES,
    ensure_experiment_layout,
    ensure_parent_dir,
    eval_metrics_name,
    finetuned_classifier_ckpt_name,
)
from dataio.marker_mapping import CLASS_LIST, LABEL_COL, MARKER_LIST
from mae.model import CellClassifier, MAEConfig, MAEEncoder


def _list_csv_files(data_dir: str, *, recursive: bool, max_files: int) -> list[str]:
    out: list[str] = []
    if recursive:
        for root, _dirs, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(".csv"):
                    out.append(os.path.join(root, f))
                    if max_files and len(out) >= max_files:
                        return sorted(out)
    else:
        for f in os.listdir(data_dir):
            if f.lower().endswith(".csv"):
                out.append(os.path.join(data_dir, f))
                if max_files and len(out) >= max_files:
                    return sorted(out)
    return sorted(out)


def _class_name(i: int) -> str:
    if 0 <= i < len(CLASS_LIST):
        return CLASS_LIST[i]
    return str(i)


def _metrics_from_confusion(conf: np.ndarray) -> dict[str, Any]:
    """conf[i,j] = 真实 i 预测为 j 的样本数。"""
    c = int(conf.shape[0])
    per_class: list[dict[str, Any]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for k in range(c):
        tp = float(conf[k, k])
        row_sum = float(conf[k].sum())
        col_sum = float(conf[:, k].sum())
        fp = max(0.0, col_sum - tp)
        fn = max(0.0, row_sum - tp)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_class.append(
            {
                "class_index": k,
                "name": _class_name(k),
                "support": int(row_sum),
                "precision": round(p, 6),
                "recall": round(r, 6),
                "f1": round(f1, 6),
            }
        )
        if row_sum > 0:
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    total = float(conf.sum())
    diag = float(np.trace(conf))
    acc = diag / total if total > 0 else 0.0
    return {
        "accuracy": round(acc, 6),
        "macro_precision": round(macro_p, 6),
        "macro_recall": round(macro_r, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class": per_class,
    }


def _write_stage2_style_eval_outputs(cm: np.ndarray, output_dir: str) -> None:
    """与 FCM ``evaluate_stage2.compute_per_class_metrics_from_confusion`` 同公式与同文件名。"""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "agg_confusion.npy"), cm)

    num_classes = cm.shape[0]
    labels = list(range(num_classes))
    total = float(cm.sum())
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

    accuracies: list[float] = []
    for i in range(num_classes):
        fn = row_sums[i] - tp[i]
        fp = col_sums[i] - tp[i]
        tn = total - (tp[i] + fn + fp)
        accuracies.append((tp[i] + tn) / total if total > 0 else 0.0)

    metrics_path = os.path.join(output_dir, "metrics_per_class.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "accuracy", "precision", "recall", "f1", "support"])
        for i in range(num_classes):
            w.writerow(
                [
                    labels[i],
                    accuracies[i],
                    precision[i],
                    recall[i],
                    f1[i],
                    int(row_sums[i]),
                ]
            )
        w.writerow(
            [
                "overall",
                float(tp.sum() / total) if total > 0 else 0.0,
                "",
                "",
                "",
                int(total),
            ]
        )
    print(f"每类指标已保存至: {metrics_path}")
    print(f"混淆矩阵已保存至: {os.path.join(output_dir, 'agg_confusion.npy')}")


def main() -> None:
    p = argparse.ArgumentParser(description="验证集评估（读取 val 目录下带标签 CSV）")
    p.add_argument("--device", type=int, default=None, help="GPU 编号；不传则自动选择")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--experiment_dir",
        type=str,
        default="",
        help="实验根目录；默认加载 finetune/finetuned_classifier_<model_size>.pt，metrics 写到 eval/metrics_<model_size>.json",
    )
    p.add_argument(
        "--model_size",
        type=str,
        default="tiny",
        choices=list(MODEL_SIZE_NAMES),
        help="须与预训练/微调 --model_size 一致",
    )
    p.add_argument(
        "--classifier_path",
        type=str,
        default="",
        help="微调分类器 checkpoint；不设且指定 --experiment_dir 则为 finetuned_classifier_<model_size>.pt",
    )
    p.add_argument(
        "--fold_dir",
        type=str,
        default="",
        help="数据折目录：未指定 --val_dir 时验证 CSV 目录为 <fold_dir>/val。",
    )
    p.add_argument("--val_dir", type=str, default="", help="验证集 CSV 目录（含标签列）")
    p.add_argument("--label_col", type=str, default="", help="标签列名；默认 marker_mapping.LABEL_COL")
    p.add_argument("--csv_recursive", action="store_true")
    p.add_argument("--csv_max_files", type=int, default=0)
    p.add_argument(
        "--metrics_json",
        type=str,
        default="",
        help="指标 JSON 路径；不设且指定 --experiment_dir 则为 <实验目录>/eval/metrics.json",
    )
    p.add_argument(
        "--predictions_csv",
        type=str,
        default="",
        help="可选：逐样本写出 真值/预测 到 CSV（不设则默认写到 eval/predictions.csv）",
    )
    p.add_argument("--write_predictions", action="store_true", help="写出 predictions_csv（默认不写，数据量大时省空间）")
    args = p.parse_args()

    val_dir = args.val_dir.strip()
    if not val_dir and args.fold_dir.strip():
        val_dir = os.path.join(args.fold_dir.strip(), "val")
    if not val_dir:
        raise ValueError("请提供 --val_dir，或 --fold_dir（将使用 fold_dir/val）")

    layout = None
    if args.experiment_dir:
        layout = ensure_experiment_layout(args.experiment_dir)
        if not args.classifier_path:
            args.classifier_path = os.path.join(
                layout["finetune"], finetuned_classifier_ckpt_name(args.model_size)
            )
        if not args.metrics_json:
            args.metrics_json = os.path.join(layout["eval"], eval_metrics_name(args.model_size))
    else:
        if not args.classifier_path:
            raise ValueError("请提供 --classifier_path，或 --experiment_dir")
        if not args.metrics_json:
            args.metrics_json = "eval_metrics.json"

    ensure_parent_dir(args.metrics_json)
    pred_path = ""
    if args.write_predictions:
        if args.predictions_csv:
            pred_path = args.predictions_csv
        elif layout is not None:
            pred_path = os.path.join(layout["eval"], "predictions.csv")
        else:
            pred_path = "predictions.csv"
        ensure_parent_dir(pred_path)

    if args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"验证 CSV 目录: {os.path.abspath(val_dir)}")

    ckpt = torch.load(args.classifier_path, map_location="cpu")
    cfg = MAEConfig(**ckpt["cfg"])
    num_classes: int = int(ckpt["num_classes"])
    encoder = MAEEncoder(cfg)
    clf = CellClassifier(encoder, num_classes=num_classes)
    clf.load_state_dict(ckpt["state_dict"])
    clf.to(device)
    clf.eval()

    label_col = args.label_col.strip() if args.label_col else LABEL_COL
    if not label_col:
        raise ValueError("请提供 --label_col 或在 marker_mapping 中设置 LABEL_COL")

    marker_list = list(MARKER_LIST)
    if len(marker_list) != cfg.seq_len:
        raise ValueError(f"marker_list 长度({len(marker_list)}) 与预训练 seq_len({cfg.seq_len}) 不一致")

    csv_files = _list_csv_files(val_dir, recursive=bool(args.csv_recursive), max_files=args.csv_max_files)
    if not csv_files:
        raise ValueError(f"在 {val_dir} 下未找到 CSV")

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_total = 0
    n_correct = 0
    pred_file = None
    pred_writer = None
    if args.write_predictions and pred_path:
        pred_file = open(pred_path, "w", newline="", encoding="utf-8")
        pred_writer = csv.writer(pred_file)
        pred_writer.writerow(["file", "row_index", "y_true", "y_pred", "y_true_name", "y_pred_name"])

    with torch.no_grad():
        for fp in tqdm(csv_files, desc="eval files", dynamic_ncols=True):
            ds = LabeledFlowCytometryDataset(
                data_path=fp,
                marker_columns=marker_list,
                label_column=label_col,
                xuhao_column="",
            )
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
            for batch in dl:
                m = batch["marker_indices"].to(device)
                v = batch["marker_values"].to(device)
                y = batch["label"].to(device).view(-1)
                logits = clf(m, v)
                pred = logits.argmax(dim=-1)
                yt = y.cpu().numpy().astype(np.int64)
                yp = pred.cpu().numpy().astype(np.int64)
                for t, q in zip(yt, yp):
                    if 0 <= t < num_classes and 0 <= q < num_classes:
                        conf[t, q] += 1
                n_total += int(yt.size)
                n_correct += int((yt == yp).sum())
                if pred_writer is not None:
                    row_idx = batch["file_row_idx"]
                    if isinstance(row_idx, torch.Tensor):
                        row_idx = row_idx.cpu().numpy()
                    base = os.path.basename(fp)
                    for i in range(yt.shape[0]):
                        ti, qi = int(yt[i]), int(yp[i])
                        ri = int(row_idx[i]) if hasattr(row_idx, "__getitem__") else i
                        pred_writer.writerow(
                            [base, ri, ti, qi, _class_name(ti), _class_name(qi)]
                        )

    if pred_file is not None:
        pred_file.close()
        print(f"已保存逐样本预测：{pred_path}")

    metrics = _metrics_from_confusion(conf)
    metrics.update(
        {
            "num_samples": n_total,
            "num_classes": num_classes,
            "val_dir": os.path.abspath(val_dir),
            "classifier_path": os.path.abspath(args.classifier_path),
            "confusion_matrix": conf.tolist(),
        }
    )
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    _write_stage2_style_eval_outputs(conf, os.path.dirname(os.path.abspath(args.metrics_json)))
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} n={n_total}")
    print(f"已保存指标：{args.metrics_json}")


if __name__ == "__main__":
    main()
