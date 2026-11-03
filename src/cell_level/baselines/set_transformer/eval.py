from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from dataio.class_lists import macro_label_indices
from cell_level.baselines.set_transformer.helpers import (
    iter_csv_chunks,
    load_checkpoint_model,
    read_split_paths,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="评估 standalone Set Transformer")
    p.add_argument("--model_path", required=True)
    p.add_argument("--test_xlsx", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--chunk_size", type=int, default=32768)
    p.add_argument("--device", type=int, default=0)
    return p.parse_args()


def write_metrics(cm: np.ndarray, output_dir: str, class_list: list[str], dataset_key: str) -> None:
    total = int(cm.sum())
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    accuracy = []
    for i in range(len(class_list)):
        fp = predicted[i] - tp[i]
        fn = support[i] - tp[i]
        tn = total - tp[i] - fp - fn
        accuracy.append((tp[i] + tn) / total if total else 0.0)
    macro_idx = macro_label_indices(class_list, dataset_key, supports=support.astype(int).tolist())
    frame = pd.DataFrame(
        {
            "class_name": class_list,
            "class_index": range(len(class_list)),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    frame.loc[len(frame)] = {
        "class_name": "overall",
        "class_index": "",
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "precision": float(np.mean(precision[macro_idx])) if macro_idx else 0.0,
        "recall": float(np.mean(recall[macro_idx])) if macro_idx else 0.0,
        "f1": float(np.mean(f1[macro_idx])) if macro_idx else 0.0,
        "support": total,
    }
    frame.to_csv(os.path.join(output_dir, "metrics_per_class.csv"), index=False)

    for normalized, name in ((False, "confusion_matrix_counts.png"), (True, "confusion_matrix_normalized.png")):
        values = cm.astype(float)
        if normalized:
            values = np.divide(values, support[:, None], out=np.zeros_like(values), where=support[:, None] != 0)
        size = max(8, min(22, len(class_list) * 0.7))
        plt.figure(figsize=(size, size))
        plt.imshow(values, cmap="Blues", aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(class_list)), class_list, rotation=45, ha="right", fontsize=7)
        plt.yticks(range(len(class_list)), class_list, fontsize=7)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, name), dpi=220)
        plt.close()


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    model, checkpoint = load_checkpoint_model(args.model_path, device)
    class_list = list(checkpoint["class_list"])
    class_to_index = {str(k): int(v) for k, v in checkpoint["class_to_index"].items()}
    marker_columns = list(checkpoint["marker_columns"])
    dataset_key = str(checkpoint["dataset_key"])
    paths = read_split_paths(args.test_xlsx, "test")
    cm = np.zeros((len(class_list), len(class_list)), dtype=np.int64)
    forward_seconds = 0.0
    cells = 0
    with torch.inference_mode():
        for file_idx, path in enumerate(paths, 1):
            file_cells = 0
            for x_np, y_np in iter_csv_chunks(
                path,
                marker_columns=marker_columns,
                class_to_index=class_to_index,
                chunksize=args.chunk_size,
            ):
                x = torch.from_numpy(x_np).unsqueeze(0).to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    pred = model(x).squeeze(0).argmax(dim=-1)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                forward_seconds += time.perf_counter() - started
                pred_np = pred.cpu().numpy()
                cm += confusion_matrix(y_np, pred_np, labels=range(len(class_list)))
                file_cells += len(y_np)
                cells += len(y_np)
            print(f"评估 {file_idx}/{len(paths)}: {os.path.basename(path)} cells={file_cells:,}", flush=True)
    np.save(os.path.join(args.output_dir, "agg_confusion.npy"), cm)
    write_metrics(cm, args.output_dir, class_list, dataset_key)
    write_json(
        os.path.join(args.output_dir, "timing.json"),
        {
            "model": "standalone_set_transformer",
            "checkpoint": os.path.abspath(args.model_path),
            "device": str(device),
            "cells": cells,
            "forward_seconds_total": forward_seconds,
            "forward_seconds_per_cell": forward_seconds / cells if cells else None,
        },
    )
    print(f"评估完成: cells={cells:,}, output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

