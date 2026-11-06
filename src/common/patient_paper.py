from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from common.patient_baselines import read_split_book


EXCLUDE_NUMERIC_COLS = {"event_id", "event", "index"}


def tube_id_from_path(path: str) -> str:
    name = Path(path).name
    match = re.search(r"tube[-_]?(\d+)", name, flags=re.IGNORECASE)
    return match.group(1) if match else "panel"


def _explicit_tube_id(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, Real) and not isinstance(value, bool) and float(value).is_integer():
        return str(int(value))
    return str(value).strip() or None


@lru_cache(maxsize=512)
def marker_frame(path: str, max_events: int) -> pd.DataFrame:
    frame = pd.read_csv(path, nrows=max_events, low_memory=False)
    numeric = frame.select_dtypes(include=[np.number]).copy()
    keep = [c for c in numeric.columns if str(c).lower() not in EXCLUDE_NUMERIC_COLS]
    numeric = numeric[keep]
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if numeric.empty:
        raise ValueError(f"No marker columns in {path}")
    return numeric


def scaled_values(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=np.float32, copy=False)
    lo = np.nanquantile(values, 0.01, axis=0)
    hi = np.nanquantile(values, 0.99, axis=0)
    return np.clip((values - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def hist2d(frame: pd.DataFrame, x_col: str, y_col: str, bins: int) -> np.ndarray:
    values = scaled_values(frame, [x_col, y_col])
    hist, _, _ = np.histogram2d(values[:, 0], values[:, 1], bins=bins, range=((0, 1), (0, 1)))
    hist = hist.astype(np.float32)
    return hist / (hist.sum() + 1e-8)


def patient_table(split_xlsx: str, split: str) -> pd.DataFrame:
    rows = read_split_book(split_xlsx, split)
    rows = rows.copy()
    rows["_tube_id"] = rows["_csv_path"].map(tube_id_from_path)
    tube_col = next((c for c in rows.columns if str(c).strip().lower() == "tube_type"), None)
    if tube_col is not None:
        explicit = rows[tube_col].map(_explicit_tube_id).astype("string")
        present = explicit.notna() & explicit.ne("")
        rows.loc[present, "_tube_id"] = explicit[present].astype(str)
    return rows


def patient_labels(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.groupby("_patient_id", sort=False)["_label"].agg(lambda s: s.mode().iat[0]).reset_index()
    return out.rename(columns={"_patient_id": "patient_id", "_label": "label"})


def split_cnn_rf_patients(labels: pd.DataFrame, seed: int = 42) -> tuple[list[str], list[str]]:
    y = labels["label"].astype(str)
    stratify = y if y.value_counts().min() >= 2 else None
    cnn_ids, rf_ids = train_test_split(
        labels["patient_id"].astype(str).to_numpy(),
        test_size=0.5,
        random_state=seed,
        stratify=stratify,
    )
    return cnn_ids.tolist(), rf_ids.tolist()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.from_numpy(images[:, None, :, :].astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


class SmallHistogramCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


def train_histogram_cnn(
    images: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float = 1e-3,
    in_channels: int = 1,
) -> SmallHistogramCNN:
    model = SmallHistogramCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(ImageDataset(images, y), batch_size=batch_size, shuffle=True, num_workers=0)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def predict_proba_cnn(model: nn.Module, images: np.ndarray, device: str, batch_size: int = 128) -> np.ndarray:
    loader = DataLoader(ImageDataset(images, np.zeros(len(images), dtype=np.int64)), batch_size=batch_size, shuffle=False)
    parts = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            parts.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(parts)


def save_predictions(output_dir: Path, ids, y_true, pred, proba, classes) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"patient_id": ids, "label": y_true, "prediction": pred})
    for i, label in enumerate(classes):
        frame[f"prob_{label}"] = proba[:, i]
    frame.to_csv(output_dir / "test_predictions.csv", index=False)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "report": classification_report(y_true, pred, output_dict=True, zero_division=0),
    }
    write_json(output_dir / "test_predictions.metrics.json", metrics)
    return metrics
