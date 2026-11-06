"""DeepCyTOF 基线：简单 MLP（交集特征）。"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from dataio.baseline_loaders import save_metrics_per_class


class DeepCyTOFMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_dims: tuple[int, ...] = (128, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = n_features
        for d in hidden_dims:
            layers += [nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(0.2)]
            d_in = d
        layers.append(nn.Linear(d_in, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_eval_deepcytof(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str,
    *,
    class_list: List[str],
    epochs: int = 50,
    batch_size: int = 4096,
    lr: float = 1e-4,
    hidden_dims: tuple[int, ...] = (128, 64),
) -> float:
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(class_list)
    labels = list(range(n_classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train.shape[1]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_s),
        torch.from_numpy(y_train).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)

    model = DeepCyTOFMLP(n_features, n_classes, hidden_dims).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("训练 DeepCyTOF (MLP)...")
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/{epochs} loss={total_loss/len(train_loader):.4f}")

    model.eval()
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_s)),
        batch_size=batch_size,
        shuffle=False,
    )
    pred_parts: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in val_loader:
            logits = model(xb.to(device))
            pred_parts.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(pred_parts)

    f1_w = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_val, y_pred, average="macro", zero_division=0)
    print(f"DeepCyTOF weighted F1: {f1_w:.4f}  macro F1: {f1_m:.4f}")
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
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_std": scaler.scale_,
            "n_features": n_features,
            "n_classes": n_classes,
            "class_list": class_list,
        },
        os.path.join(output_dir, "model.pt"),
    )
    print(f"DeepCyTOF 结果已保存至 {output_dir}")
    return f1_w
