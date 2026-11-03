"""逻辑回归基线（sklearn，交集特征）。"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from dataio.baseline_loaders import save_metrics_per_class


def train_eval_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str,
    *,
    class_list: List[str],
    max_iter: int = 500,
    C: float = 1.0,
) -> float:
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(class_list)
    labels = list(range(n_classes))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    print("训练逻辑回归...")
    clf = LogisticRegression(
        max_iter=max_iter,
        C=C,
        solver="lbfgs",
        random_state=42,
        verbose=0,
    )
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_val_s)
    f1_w = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_val, y_pred, average="macro", zero_division=0)
    print(f"逻辑回归 weighted F1: {f1_w:.4f}  macro F1: {f1_m:.4f}")
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
    np.savez(
        os.path.join(output_dir, "model.npz"),
        coef=clf.coef_,
        intercept=clf.intercept_,
        classes=clf.classes_,
        scale_mean=scaler.mean_,
        scale_std=scaler.scale_,
        class_list=np.array(class_list, dtype=object),
    )
    print(f"逻辑回归结果已保存至 {output_dir}")
    return f1_w
