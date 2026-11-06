from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataio.class_lists import (
    dataset_key_from_split_xlsx,
    load_dataset_class_info,
    macro_label_indices,
)
from dataio.flow_dataset import _read_csv_with_fallback
from dataio.marker_mapping import LABEL_COLUMN, MARKER_LIST, MARKER_TO_INDEX
from dataio.paths import remap_manifest_path
from cell_level.baselines.set_transformer.model import StandaloneSetTransformer




def read_split_paths(xlsx_path: str, split: str, csv_col: str = "csv_path") -> list[str]:
    df = pd.read_excel(xlsx_path, sheet_name=split)
    if csv_col not in df.columns:
        raise KeyError(f"{xlsx_path} 的 {split} 子表缺少 {csv_col}: {list(df.columns)}")
    paths: list[str] = []
    missing = 0
    for raw in df[csv_col].dropna().astype(str):
        path = os.path.abspath(os.path.expanduser(remap_manifest_path(raw.strip())))
        if os.path.isfile(path):
            paths.append(path)
        else:
            missing += 1
    if missing:
        print(f"警告: {split} 子表有 {missing} 个 CSV 不存在，已跳过", flush=True)
    if not paths:
        raise RuntimeError(f"{xlsx_path} 的 {split} 子表没有可用 CSV")
    return paths


def infer_marker_columns(paths: Sequence[str], label_column: str = LABEL_COLUMN) -> list[str]:
    found: set[str] = set()
    ignore = {label_column, "Time", "event_ID"}
    for i, path in enumerate(paths, 1):
        columns = list(_read_csv_with_fallback(path, nrows=0).columns)
        found.update(c for c in columns if c not in ignore and c in MARKER_TO_INDEX)
        if i == 1 or i % 100 == 0 or i == len(paths):
            print(f"扫描 marker 表头: {i}/{len(paths)}", flush=True)
    ordered = [name for name in MARKER_LIST if name in found]
    if not ordered:
        raise RuntimeError("没有找到 MARKER_TO_INDEX 中已登记的特征列")
    return ordered


def iter_csv_chunks(
    csv_path: str,
    *,
    marker_columns: Sequence[str],
    class_to_index: dict[str, int],
    label_column: str = LABEL_COLUMN,
    chunksize: int = 65536,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    header = list(_read_csv_with_fallback(csv_path, nrows=0).columns)
    available = [c for c in marker_columns if c in header]
    if label_column not in header:
        raise ValueError(f"{csv_path} 缺少标签列 {label_column}")
    usecols = available + [label_column]
    marker_pos = {name: i for i, name in enumerate(marker_columns)}
    reader = _read_csv_with_fallback(
        csv_path,
        usecols=usecols,
        dtype={label_column: str},
        chunksize=chunksize,
    )
    for frame in reader:
        x = np.zeros((len(frame), len(marker_columns)), dtype=np.float32)
        if available:
            values = frame[available].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            x[:, [marker_pos[c] for c in available]] = values
        labels = frame[label_column].astype(str).str.strip()
        y = labels.map(class_to_index).fillna(0).to_numpy(np.int64)
        yield x, y


def _cache_stem(csv_path: str, configuration: dict) -> str:
    """Identify the complete source and configuration, including marker order."""
    digest = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    digest.update(b"\0")
    with open(csv_path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    safe = Path(csv_path).stem.replace(" ", "_")[:80]
    return f"{safe}_{digest.hexdigest()}"


def build_uniform_tube_cache(
    csv_path: str,
    *,
    cache_dir: str,
    marker_columns: Sequence[str],
    class_to_index: dict[str, int],
    max_cells: int,
    seed: int,
    chunksize: int = 65536,
) -> tuple[str, str, int]:
    """Create a deterministic uniform reservoir sample without loading a whole tube."""
    os.makedirs(cache_dir, exist_ok=True)
    stem = _cache_stem(
        csv_path,
        {
            "version": 1,
            "csv_path": csv_path,
            "marker_columns": list(marker_columns),
            "class_to_index": class_to_index,
            "max_cells": max_cells,
            "seed": seed,
            "chunksize": chunksize,
        },
    )
    x_path = os.path.join(cache_dir, stem + "_x.npy")
    y_path = os.path.join(cache_dir, stem + "_y.npy")
    if os.path.isfile(x_path) and os.path.isfile(y_path):
        try:
            x_cached = np.load(x_path, mmap_mode="r")
            y_cached = np.load(y_path, mmap_mode="r")
            if (
                x_cached.ndim == 2
                and x_cached.shape[1] == len(marker_columns)
                and x_cached.dtype == np.float32
                and y_cached.shape == (x_cached.shape[0],)
                and y_cached.dtype == np.int64
                and x_cached.shape[0] <= max_cells
            ):
                return x_path, y_path, int(x_cached.shape[0])
        except (OSError, ValueError, EOFError):
            # Interrupted or malformed cache writes can be rebuilt from the CSV.
            pass

    rng_seed = seed ^ int(hashlib.sha1(csv_path.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(rng_seed)
    kept_x = np.empty((0, len(marker_columns)), dtype=np.float32)
    kept_y = np.empty((0,), dtype=np.int64)
    kept_keys = np.empty((0,), dtype=np.float64)
    total = 0
    for x, y in iter_csv_chunks(
        csv_path,
        marker_columns=marker_columns,
        class_to_index=class_to_index,
        chunksize=chunksize,
    ):
        total += len(y)
        keys = rng.random(len(y))
        all_x = np.concatenate((kept_x, x), axis=0)
        all_y = np.concatenate((kept_y, y), axis=0)
        all_keys = np.concatenate((kept_keys, keys), axis=0)
        if len(all_keys) > max_cells:
            idx = np.argpartition(all_keys, max_cells - 1)[:max_cells]
            kept_x, kept_y, kept_keys = all_x[idx], all_y[idx], all_keys[idx]
        else:
            kept_x, kept_y, kept_keys = all_x, all_y, all_keys
    np.save(x_path, kept_x)
    np.save(y_path, kept_y)
    print(
        f"缓存 {os.path.basename(csv_path)}: 原始 {total:,} -> 集合 {len(kept_y):,}",
        flush=True,
    )
    return x_path, y_path, int(len(kept_y))


def load_checkpoint_model(model_path: str, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = StandaloneSetTransformer(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def class_info(xlsx_path: str, mapping_path: str):
    key = dataset_key_from_split_xlsx(xlsx_path)
    return load_dataset_class_info(mapping_path, dataset_key=key)


def macro_f1_from_confusion(cm: np.ndarray, class_list: list[str], dataset_key: str) -> float:
    support = cm.sum(axis=1).astype(int).tolist()
    labels = macro_label_indices(class_list, dataset_key, supports=support)
    values: list[float] = []
    for i in labels:
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        denom = 2 * tp + fp + fn
        values.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(values)) if values else 0.0

