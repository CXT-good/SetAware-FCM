#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.flow_dataset import TubeDataset
from cell_level.pactransformer.model import SetTransformer


def parse_args():
    p = argparse.ArgumentParser(description="Stage2 formal evaluation with predictions/probabilities and full metrics")
    p.add_argument("--model_path", required=True)
    p.add_argument("--embeddings_dir", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--xuhao_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap_max_n", type=int, default=200000)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--timing_warmup_batches", type=int, default=5)
    return p.parse_args()


def load_model(path: str, num_classes_arg, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    num_classes = int(num_classes_arg or ckpt.get("num_classes", 2))
    model = SetTransformer(
        dim_input=int(ckpt.get("dim_input", 128)),
        dim_hidden=int(ckpt.get("dim_hidden", 32)),
        num_heads=int(ckpt.get("num_heads", 4)),
        num_inds=int(ckpt.get("num_inds", 16)),
        num_layers=int(ckpt.get("num_layers", 3)),
        num_classes=num_classes,
        dropout=float(ckpt.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt, num_classes


def flatten_xuhao(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().reshape(-1)
    arr = np.asarray(x)
    return arr.reshape(-1)


def bootstrap_ci_binary(y_true, score, metric_fn, n_boot=1000, max_n=200000, seed=2024):
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    valid = np.isfinite(score)
    y_true, score = y_true[valid], score[valid]
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    if len(y_true) > max_n:
        idx0 = np.where(y_true == 0)[0]
        idx1 = np.where(y_true == 1)[0]
        n1 = min(len(idx1), max(1000, int(max_n * len(idx1) / len(y_true))))
        n0 = min(len(idx0), max_n - n1)
        keep = np.concatenate([rng.choice(idx0, n0, replace=False), rng.choice(idx1, n1, replace=False)])
        y_true, score = y_true[keep], score[keep]
    point = metric_fn(y_true, score)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y_true[idx]
        if len(np.unique(yy)) < 2:
            continue
        try:
            vals.append(metric_fn(yy, score[idx]))
        except Exception:
            pass
    if not vals:
        return float(point), np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def safe_prauc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, score))


def compute_metrics(y_true, y_pred, probs, num_classes, bootstrap, bootstrap_max_n, seed):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probs = np.asarray(probs, dtype=float)
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    rows = []
    for c in labels:
        yy = (y_true == c).astype(int)
        pp = (y_pred == c).astype(int)
        tn, fp, fn, tp = confusion_matrix(yy, pp, labels=[0, 1]).ravel()
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        f1 = 2 * ppv * sens / max(ppv + sens, 1e-12)
        bacc = 0.5 * (sens + spec)
        score = probs[:, c] if probs.ndim == 2 and probs.shape[1] > c else pp.astype(float)
        auc, auc_lo, auc_hi = bootstrap_ci_binary(yy, score, safe_auc, bootstrap, bootstrap_max_n, seed + c * 17)
        prauc, prauc_lo, prauc_hi = bootstrap_ci_binary(yy, score, safe_prauc, bootstrap, bootstrap_max_n, seed + c * 31)
        rows.append({
            "label": c,
            "accuracy": acc,
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "precision": ppv,
            "recall": sens,
            "f1": f1,
            "bacc": bacc,
            "auc": auc,
            "auc_ci_low": auc_lo,
            "auc_ci_high": auc_hi,
            "prauc": prauc,
            "prauc_ci_low": prauc_lo,
            "prauc_ci_high": prauc_hi,
            "support": int((y_true == c).sum()),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })
    per_class = pd.DataFrame(rows)
    if num_classes == 2:
        pos = per_class.loc[per_class["label"] == 1].iloc[0].to_dict()
        overall = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "sensitivity": pos["sensitivity"],
            "specificity": pos["specificity"],
            "ppv": pos["ppv"],
            "npv": pos["npv"],
            "f1": pos["f1"],
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "bacc": float(balanced_accuracy_score(y_true, y_pred)),
            "auc": pos["auc"], "auc_ci_low": pos["auc_ci_low"], "auc_ci_high": pos["auc_ci_high"],
            "prauc": pos["prauc"], "prauc_ci_low": pos["prauc_ci_low"], "prauc_ci_high": pos["prauc_ci_high"],
            "support": int(total),
        }
    else:
        overall = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "mean_sensitivity": float(per_class["sensitivity"].mean()),
            "mean_specificity": float(per_class["specificity"].mean()),
            "mean_ppv": float(per_class["ppv"].mean()),
            "mean_npv": float(per_class["npv"].mean()),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "mean_bacc": float(per_class["bacc"].mean()),
            "mean_auc": float(per_class["auc"].mean()),
            "mean_prauc": float(per_class["prauc"].mean()),
            "support": int(total),
        }
    return cm, per_class, pd.DataFrame([overall])


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model, ckpt, num_classes = load_model(args.model_path, args.num_classes, device)
    dataset = TubeDataset(args.embeddings_dir, args.labels_dir, args.xuhao_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    pred_path = out / "predictions.csv.gz"
    prob_path = out / "probabilities.csv.gz"
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    prob_all: List[np.ndarray] = []
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    timing = {"device": str(device), "model_path": args.model_path, "timed_batches": 0, "timed_samples": 0, "forward_seconds_total": 0.0}

    with gzip.open(pred_path, "wt", newline="", encoding="utf-8", compresslevel=6) as pf, gzip.open(prob_path, "wt", newline="", encoding="utf-8", compresslevel=6) as qf:
        pred_writer = csv.writer(pf)
        prob_writer = csv.writer(qf)
        pred_writer.writerow(["tube_id", "xuhao", "true_label", "pred_label"])
        prob_writer.writerow(["tube_id", "xuhao", "true_label", "pred_label"] + [f"prob_{i}" for i in range(num_classes)])
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc="Stage2 formal eval")):
                x = batch["embeddings"].to(device)
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                y = batch["labels"].to(device).long()
                xuhao = flatten_xuhao(batch["xuhao"])
                tube_id = dataset.tube_ids[batch_idx]
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                logits = model(x)
                if device.type == "cuda": torch.cuda.synchronize()
                t1 = time.perf_counter()
                if batch_idx + 1 > args.timing_warmup_batches:
                    timing["timed_batches"] += 1
                    timing["timed_samples"] += int(x.shape[1])
                    timing["forward_seconds_total"] += float(t1 - t0)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1, num_classes)
                pred = probs.argmax(axis=1).astype(int)
                true = y.detach().cpu().numpy().reshape(-1).astype(int)
                cm += confusion_matrix(true, pred, labels=list(range(num_classes)))
                y_true_all.append(true); y_pred_all.append(pred); prob_all.append(probs)
                rows_pred = zip([tube_id]*len(true), xuhao.tolist(), true.tolist(), pred.tolist())
                pred_writer.writerows(rows_pred)
                rows_prob = ([tube_id, xuhao[i], int(true[i]), int(pred[i]), *[float(v) for v in probs[i]]] for i in range(len(true)))
                prob_writer.writerows(rows_prob)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    probs = np.vstack(prob_all)
    cm, per_class, overall = compute_metrics(y_true, y_pred, probs, num_classes, args.bootstrap, args.bootstrap_max_n, args.seed)
    np.save(out / "agg_confusion.npy", cm)
    per_class.to_csv(out / "metrics_per_class.csv", index=False)
    overall.to_csv(out / "overall_metrics.csv", index=False)
    with (out / "inference_timing.json").open("w", encoding="utf-8") as f:
        timing["forward_seconds_per_sample"] = timing["forward_seconds_total"] / timing["timed_samples"] if timing["timed_samples"] else None
        json.dump(timing, f, indent=2, ensure_ascii=False)
    with (out / "eval_meta.json").open("w", encoding="utf-8") as f:
        json.dump({"checkpoint_meta": {k: v for k, v in ckpt.items() if k != "model_state_dict" and k != "optimizer_state_dict"}, "num_classes": num_classes}, f, indent=2, ensure_ascii=False)
    print(f"saved formal stage2 eval to {out}")


if __name__ == "__main__":
    main()
