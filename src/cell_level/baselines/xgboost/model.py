"""XGBoost 基线（并集特征）。"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from dataio.baseline_loaders import save_metrics_per_class


def train_eval_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str,
    *,
    class_list: List[str],
    n_estimators: int = 200,
    max_depth: int = 8,
) -> float:
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("请安装 xgboost: pip install xgboost") from e

    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(class_list)
    labels = list(range(n_classes))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    dtrain = xgb.DMatrix(X_train_s, label=y_train)
    dval = xgb.DMatrix(X_val_s, label=y_val)
    params = {
        "objective": "multi:softmax",
        "num_class": n_classes,
        "max_depth": max_depth,
        "eta": 0.1,
        "seed": 42,
    }
    print("训练 XGBoost...")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=False,
    )
    y_pred = bst.predict(dval).astype(np.int64)
    f1_w = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_val, y_pred, average="macro", zero_division=0)
    print(f"XGBoost weighted F1: {f1_w:.4f}  macro F1: {f1_m:.4f}")
    report = classification_report(
        y_val, y_pred, labels=labels, target_names=class_list, zero_division=0
    )
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    pd.DataFrame(cm, index=class_list, columns=class_list).to_csv(
        os.path.join(output_dir, "confusion_matrix.csv")
    )
    save_metrics_per_class(
        cm, os.path.join(output_dir, "metrics_per_class.csv"), class_names=class_list
    )
    with open(os.path.join(output_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"weighted_f1: {f1_w:.4f}\nmacro_f1: {f1_m:.4f}\n\n")
        f.write(report)
    bst.save_model(os.path.join(output_dir, "model.json"))
    np.savez(
        os.path.join(output_dir, "scaler.npz"),
        mean=scaler.mean_,
        std=scaler.scale_,
        class_list=np.array(class_list, dtype=object),
    )
    print(f"XGBoost 结果已保存至 {output_dir}")
    return f1_w
