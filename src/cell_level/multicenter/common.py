from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from dataio.flow_dataset import _read_csv_with_fallback
from dataio.paths import remap_manifest_path


COMMON_MARKERS: tuple[str, ...] = (
    "CD10",
    "CD19",
    "CD20",
    "CD34",
    "CD38",
    "CD45",
    "FSC-A",
    "FSC-W",
    "SSC-A",
    "SY41",
)

CENTER_XLSX: dict[str, str] = {
    "bln": "data/output_flowrepository_plus_gates_transformed_Bln_dataset_split.xlsx",
    "bue": "data/output_flowrepository_plus_gates_transformed_Bue_dataset_split.xlsx",
    "vie14": "data/output_flowrepository_plus_gates_transformed_Routine_dataset_split.xlsx",
}


def read_split_paths(split_xlsx: str, sheet: str) -> list[str]:
    frame = pd.read_excel(split_xlsx, sheet_name=sheet)
    if "csv_path" not in frame.columns:
        raise ValueError(f"csv_path missing in {split_xlsx}:{sheet}")
    candidates = [remap_manifest_path(str(raw).strip()) for raw in frame["csv_path"]]
    paths = [str(Path(path).resolve()) for path in candidates if path.lower().endswith(".csv") and Path(path).is_file()]
    if not paths:
        raise ValueError(f"no readable CSV files in {split_xlsx}:{sheet}")
    return paths


def iter_csv_chunks(
    csv_path: str,
    marker_columns: Sequence[str],
    class_to_index: dict[str, int],
    *,
    label_column: str = "label",
    chunksize: int = 200_000,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    header = _read_csv_with_fallback(csv_path, nrows=0).columns.tolist()
    missing = [marker for marker in marker_columns if marker not in header]
    if missing:
        raise ValueError(f"missing required markers {missing}: {csv_path}")
    if label_column not in header:
        raise ValueError(f"missing label column {label_column}: {csv_path}")

    required = list(marker_columns) + [label_column]
    reader = _read_csv_with_fallback(
        csv_path,
        usecols=required,
        dtype={label_column: str},
        chunksize=chunksize,
    )
    for frame in reader:
        labels = frame[label_column].fillna("").astype(str).str.strip()
        unknown = sorted(
            value for value in labels.unique() if value and value not in class_to_index
        )
        if unknown:
            raise ValueError(f"unknown labels {unknown}: {csv_path}")
        x = frame[list(marker_columns)].apply(pd.to_numeric, errors="coerce")
        x_np = np.nan_to_num(
            x.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        y_np = labels.map(class_to_index).fillna(0).to_numpy(dtype=np.int64)
        yield x_np, y_np


def load_source_training_data(
    split_xlsx: str,
    class_list: Sequence[str],
    *,
    max_cells_per_file: int = 50_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    paths = read_split_paths(split_xlsx, "train")
    class_to_index = {str(name): index for index, name in enumerate(class_list)}
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for path in paths:
        chunks = list(
            iter_csv_chunks(path, COMMON_MARKERS, class_to_index, chunksize=200_000)
        )
        x = np.concatenate([part[0] for part in chunks])
        y = np.concatenate([part[1] for part in chunks])
        if max_cells_per_file > 0 and len(y) > max_cells_per_file:
            path_seed = int.from_bytes(Path(path).name.encode("utf-8"), "little") % (2**32)
            rng = np.random.default_rng(seed ^ path_seed)
            chosen = rng.choice(len(y), max_cells_per_file, replace=False)
            x, y = x[chosen], y[chosen]
        all_x.append(x)
        all_y.append(y)
    return np.concatenate(all_x), np.concatenate(all_y), paths


def metrics_from_confusion(
    confusion: np.ndarray, class_names: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cm = np.asarray(confusion, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion matrix must be square")
    if cm.shape[0] != len(class_names):
        raise ValueError("class_names length does not match confusion matrix")

    total = float(cm.sum())
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    fn = support - tp
    fp = predicted - tp
    tn = total - tp - fn - fp

    def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator != 0,
        )

    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    precision = divide(tp, tp + fp)
    f1 = divide(2.0 * precision * sensitivity, precision + sensitivity)
    bacc = (sensitivity + specificity) / 2.0
    one_vs_rest_accuracy = divide(tp + tn, np.full_like(tp, total))
    per_class = pd.DataFrame(
        {
            "class_name": list(class_names),
            "class_index": np.arange(len(class_names)),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "bacc": bacc,
            "accuracy": one_vs_rest_accuracy,
            "support": support.astype(np.int64),
        }
    )
    active = support > 0
    overall = pd.DataFrame(
        [
            {
                "sensitivity": float(sensitivity[active].mean()) if active.any() else 0.0,
                "specificity": float(specificity[active].mean()) if active.any() else 0.0,
                "f1": float(f1[active].mean()) if active.any() else 0.0,
                "bacc": float(bacc[active].mean()) if active.any() else 0.0,
                "accuracy": float(tp.sum() / total) if total else 0.0,
                "n_samples": int(total),
            }
        ]
    )
    return per_class, overall


def write_metric_artifacts(
    confusion: np.ndarray,
    class_names: Sequence[str],
    output_dir: str | os.PathLike[str],
    metadata: dict,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_class, overall = metrics_from_confusion(confusion, class_names)
    np.save(out / "agg_confusion.npy", np.asarray(confusion, dtype=np.int64))
    per_class.to_csv(out / "metrics_per_class.csv", index=False, float_format="%.4f")
    overall.to_csv(out / "overall_metrics.csv", index=False, float_format="%.4f")
    (out / "eval_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
