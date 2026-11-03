"""
基线模型数据加载：
- 从 *_dataset_split.xlsx 的 train / test 子表读取 csv_path（与 stage1 一致）
- 类别映射来自 datasets_class_mappings.json（每数据集独立 class_to_index）
- label 列强制按字符串读取
- intersection：LR / DeepCyTOF；union：XGBoost（缺失 marker 填 0）
"""
from __future__ import annotations

import os
from functools import reduce
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from dataio.class_lists import dataset_key_from_split_xlsx
from dataio.paths import remap_manifest_path
from dataio.marker_mapping import (
    LABEL_COLUMN,
    MARKER_TO_INDEX,
    load_class_list_for_dataset,
)

IGNORE_COLS = {LABEL_COLUMN, "Time", "event_ID"}


def _read_csv_with_fallback(path: str, **kwargs) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"无法用常见编码解析: {path}")


def count_train_samples_for_split(
    dataset_xlsx: str,
    *,
    max_cells_per_file: Optional[int] = None,
    csv_col: str = "csv_path",
) -> int:
    """统计 train 子表加载后的训练样本数（与 load_split_xlsx 的 max_cells 截断规则一致）。"""
    total = 0
    for path in collect_csv_paths_from_xlsx(dataset_xlsx, sheet="train", csv_col=csv_col):
        with open(path, "rb") as f:
            n = max(0, sum(1 for _ in f) - 1)
        if max_cells_per_file and n > max_cells_per_file:
            n = max_cells_per_file
        total += n
    return total


def collect_csv_paths_from_xlsx(
    xlsx_path: str,
    *,
    sheet: str = "train",
    csv_col: str = "csv_path",
) -> list[str]:
    if not os.path.isabs(xlsx_path):
        raise ValueError(f"xlsx 必须是绝对路径: {xlsx_path}")
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    if csv_col not in df.columns:
        raise ValueError(f"{xlsx_path} sheet={sheet} 缺少列 {csv_col}")
    paths: list[str] = []
    missing = 0
    for raw in df[csv_col].astype(str):
        p = os.path.abspath(remap_manifest_path(raw.strip()))
        if not p.lower().endswith(".csv"):
            continue
        if os.path.isfile(p):
            paths.append(p)
        else:
            missing += 1
    if missing:
        print(f"警告: {sheet} 子表有 {missing} 个 csv_path 不存在（已跳过）")
    return paths


def _marker_columns_in_file(path: str) -> set[str]:
    df = _read_csv_with_fallback(path, nrows=0)
    candidates = [c for c in df.columns if c not in IGNORE_COLS]
    return {c for c in candidates if c in MARKER_TO_INDEX}


def compute_marker_sets_from_paths(
    csv_paths: list[str],
    *,
    file_filter: Optional[Callable[[str], bool]] = None,
) -> Tuple[List[str], List[str]]:
    """从 CSV 路径列表计算 marker 交集 / 并集（仅限 MARKER_TO_INDEX）。"""
    files = []
    for p in csv_paths:
        name = os.path.basename(p)
        if file_filter is not None and not file_filter(name):
            continue
        files.append(p)
    if not files:
        raise ValueError("未找到可用 CSV 文件")
    sets_list = [_marker_columns_in_file(f) for f in files]
    intersection = sorted(reduce(set.intersection, sets_list))
    union = sorted(reduce(set.union, sets_list))
    return intersection, union


def _resolve_label_column(path: str) -> str:
    df = _read_csv_with_fallback(path, nrows=0)
    if LABEL_COLUMN in df.columns:
        return LABEL_COLUMN
    for alt in ("中文类别名称", "中文名称"):
        if alt in df.columns:
            return alt
    raise ValueError(f"文件缺少标签列 {LABEL_COLUMN}: {path}")


def _labels_to_indices(labels, class_to_index: dict[str, int]) -> np.ndarray:
    out = []
    for v in labels:
        key = str(v).strip() if pd.notna(v) else ""
        if not key or key.lower() == "nan":
            out.append(0)
        else:
            out.append(class_to_index.get(key, 0))
    return np.array(out, dtype=np.int64)


def load_file_intersection(
    path: str,
    marker_columns: List[str],
    class_to_index: dict[str, int],
    *,
    label_column: str = LABEL_COLUMN,
    max_cells: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    label_col = _resolve_label_column(path)
    usecols = list(marker_columns) + [label_col]
    df = _read_csv_with_fallback(path, usecols=usecols, dtype={label_col: str})
    df[label_col] = df[label_col].astype(str).str.strip()
    missing = set(marker_columns) - set(df.columns)
    if missing:
        raise ValueError(f"文件缺少交集列 {missing}: {path}")
    X = df[marker_columns].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = _labels_to_indices(df[label_col].values, class_to_index)
    if max_cells and len(X) > max_cells:
        rng = rng or np.random.default_rng(42)
        idx = rng.choice(len(X), max_cells, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def load_file_union(
    path: str,
    marker_columns: List[str],
    class_to_index: dict[str, int],
    *,
    label_column: str = LABEL_COLUMN,
    max_cells: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    label_col = _resolve_label_column(path)
    df = _read_csv_with_fallback(path, dtype={label_col: str})
    df[label_col] = df[label_col].astype(str).str.strip()
    if label_col not in df.columns:
        raise ValueError(f"文件缺少标签列: {path}")
    n_markers = len(marker_columns)
    X = np.zeros((len(df), n_markers), dtype=np.float32)
    for i, col in enumerate(marker_columns):
        if col in df.columns:
            X[:, i] = np.nan_to_num(df[col].values.astype(np.float32), nan=0.0)
    y = _labels_to_indices(df[label_col].values, class_to_index)
    if max_cells and len(X) > max_cells:
        rng = rng or np.random.default_rng(42)
        idx = rng.choice(len(X), max_cells, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def load_split_xlsx(
    dataset_xlsx: str,
    mode: str,
    *,
    class_mapping_path: str,
    max_cells_per_file_train: Optional[int] = None,
    max_cells_per_file_test: Optional[int] = None,
    seed: int = 42,
    csv_col: str = "csv_path",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str], dict[str, int]]:
    """
    从同一 *_dataset_split.xlsx 加载训练 / 测试特征矩阵。
    - train 子表 -> 训练集 csv_path
    - test 子表  -> 测试集 csv_path

    Returns:
        X_train, y_train, X_test, y_test, marker_columns, class_list, class_to_index
    """
    if mode not in ("intersection", "union"):
        raise ValueError(f"mode 必须是 intersection 或 union: {mode}")

    dataset_xlsx = os.path.abspath(dataset_xlsx)
    ds_key = dataset_key_from_split_xlsx(dataset_xlsx)
    class_list, _ = load_class_list_for_dataset(ds_key, mapping_path=class_mapping_path)
    class_to_index = {c: i for i, c in enumerate(class_list)}

    train_paths = collect_csv_paths_from_xlsx(dataset_xlsx, sheet="train", csv_col=csv_col)
    test_paths = collect_csv_paths_from_xlsx(dataset_xlsx, sheet="test", csv_col=csv_col)
    if not train_paths:
        raise ValueError(f"train 子表无可用 CSV: {dataset_xlsx}")
    if not test_paths:
        raise ValueError(f"test 子表无可用 CSV: {dataset_xlsx}")

    inter, union = compute_marker_sets_from_paths(train_paths + test_paths)
    marker_columns = inter if mode == "intersection" else union
    print(f"dataset_key={ds_key}  n_classes={len(class_list)}")
    print(f"marker mode={mode}  n_markers={len(marker_columns)}")

    rng = np.random.default_rng(seed)
    load_fn = load_file_intersection if mode == "intersection" else load_file_union

    X_train_list, y_train_list = [], []
    for path in train_paths:
        try:
            X, y = load_fn(
                path,
                marker_columns,
                class_to_index,
                max_cells=max_cells_per_file_train,
                rng=rng,
            )
            if len(X):
                X_train_list.append(X)
                y_train_list.append(y)
        except Exception as e:
            print(f"  跳过 train {os.path.basename(path)}: {e}")

    X_test_list, y_test_list = [], []
    for path in test_paths:
        try:
            X, y = load_fn(
                path,
                marker_columns,
                class_to_index,
                max_cells=max_cells_per_file_test,
                rng=rng,
            )
            if len(X):
                X_test_list.append(X)
                y_test_list.append(y)
        except Exception as e:
            print(f"  跳过 test {os.path.basename(path)}: {e}")

    if not X_train_list:
        raise ValueError("训练集为空，请检查 CSV 与 marker 列")
    if not X_test_list:
        raise ValueError("测试集为空，请检查 CSV 与 marker 列")

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    return X_train, y_train, X_test, y_test, marker_columns, class_list, class_to_index


def save_metrics_per_class(
    cm: np.ndarray,
    output_path: str,
    *,
    class_names: Optional[List[str]] = None,
    dataset_key: Optional[str] = None,
) -> None:
    """从混淆矩阵保存每类指标；class_names 为类别名字符串列表。"""
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError("class_names 长度与混淆矩阵不一致")

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

    y_true, y_pred = [], []
    for i in range(num_classes):
        for j in range(num_classes):
            n = int(cm[i, j])
            if n:
                y_true.extend([i] * n)
                y_pred.extend([j] * n)

    from dataio.class_lists import macro_label_indices

    macro_labels = macro_label_indices(
        class_names, dataset_key, supports=row_sums.astype(int).tolist()
    )
    macro_f1 = macro_p = macro_r = float("nan")
    if y_true and macro_labels:
        from sklearn.metrics import f1_score, precision_score, recall_score

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
            "class_name": class_names,
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
    metrics_df.to_csv(output_path, index=False)
