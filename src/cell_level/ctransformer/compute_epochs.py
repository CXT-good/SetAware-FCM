"""
按「总训练 step 数」对齐 epoch：与 train.py 中 steps_per_epoch 统计方式一致。

基准：Zhongnan_Hospital 训练 REF_EPOCHS（默认 100）个 epoch 的总 step 数为 target。
其他数据集：epochs(D) = round(REF_EPOCHS * steps_per_epoch(REF) / steps_per_epoch(D))

即各数据集的总迭代次数（所有 epoch、所有 CSV 的 batch 数之和）大致相同。
"""

from __future__ import annotations

import argparse
import json
import math
import os

import pandas as pd
from dataio.paths import remap_manifest_path

REF_DATASET_KEY = "Zhongnan_Hospital"
REF_EPOCHS = 60
DEFAULT_BATCH_SIZE = 2048
DEFAULT_DATA_DIR = "data"
DEFAULT_CACHE_PATH = "cache/dataset_train_steps_cache.json"


def _train_csv_paths(xlsx_path: str, *, csv_col: str = "csv_path") -> list[str]:
    df = pd.read_excel(xlsx_path, sheet_name="train")
    if csv_col not in df.columns:
        raise ValueError(f"{xlsx_path} train 子表缺少列 {csv_col}")
    out: list[str] = []
    for raw in df[csv_col].astype(str):
        p = os.path.abspath(remap_manifest_path(raw.strip()))
        if p.lower().endswith(".csv") and os.path.isfile(p):
            out.append(p)
    return out


def _count_csv_rows(csv_path: str) -> int:
    with open(csv_path, "rb") as f:
        return max(0, sum(1 for _ in f) - 1)


def steps_per_epoch_for_dataset(
    dataset_key: str,
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    csv_col: str = "csv_path",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """与 train.py 的 _build_warmup_cosine_scheduler 一致：逐 CSV ceil(rows/batch) 求和。"""
    xlsx = os.path.join(data_dir, f"{dataset_key}_dataset_split.xlsx")
    if not os.path.isfile(xlsx):
        raise FileNotFoundError(f"未找到 split xlsx: {xlsx}")
    steps = 0
    for p in _train_csv_paths(xlsx, csv_col=csv_col):
        n = _count_csv_rows(p)
        if n > 0:
            steps += int(math.ceil(n / float(batch_size)))
    return steps


def build_steps_cache(
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    cache_path: str = DEFAULT_CACHE_PATH,
    csv_col: str = "csv_path",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, int]:
    steps_map: dict[str, int] = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith("_dataset_split.xlsx"):
            continue
        key = f[: -len("_dataset_split.xlsx")]
        print(f"[scan] {key} ...", flush=True)
        steps_map[key] = steps_per_epoch_for_dataset(
            key, data_dir=data_dir, csv_col=csv_col, batch_size=batch_size
        )
        print(f"  steps_per_epoch={steps_map[key]}", flush=True)

    payload = {
        "version": 2,
        "ref_dataset_key": REF_DATASET_KEY,
        "ref_epochs": REF_EPOCHS,
        "batch_size": batch_size,
        "metric": "steps_per_epoch",
        "datasets": steps_map,
    }
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return steps_map


def load_steps_cache(cache_path: str = DEFAULT_CACHE_PATH) -> dict[str, int]:
    path = os.path.abspath(cache_path)
    if not os.path.isfile(path):
        return build_steps_cache(cache_path=path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {k: int(v) for k, v in payload.get("datasets", {}).items()}


def compute_epochs_from_sample_count(
    n_train_samples: int,
    n_ref_train_samples: int,
    *,
    ref_epochs: int = REF_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_epochs: int = 1,
) -> int:
    """
    按实际训练样本数对齐总迭代次数（optimizer steps）。

    基准：Zhongnan 训练 ref_epochs 个 epoch 的总 batch 数；
    其他数据集：epochs = round(ref_epochs * steps_ref / steps_cur)。
    其中 steps = ceil(n_train / batch_size)。
    """
    steps_per_epoch = max(1, math.ceil(n_train_samples / float(batch_size)))
    ref_steps_per_epoch = max(1, math.ceil(n_ref_train_samples / float(batch_size)))
    target_total_steps = ref_epochs * ref_steps_per_epoch
    epochs = round(target_total_steps / steps_per_epoch)
    return max(min_epochs, int(epochs))


def compute_epochs(
    dataset_key: str,
    *,
    ref_dataset_key: str = REF_DATASET_KEY,
    ref_epochs: int = REF_EPOCHS,
    steps_map: dict[str, int] | None = None,
    cache_path: str = DEFAULT_CACHE_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_epochs: int = 1,
) -> int:
    if steps_map is None:
        steps_map = load_steps_cache(cache_path)
    if ref_dataset_key not in steps_map:
        raise KeyError(f"缓存中无基准数据集: {ref_dataset_key}")
    if dataset_key not in steps_map:
        steps_map[dataset_key] = steps_per_epoch_for_dataset(
            dataset_key, batch_size=batch_size
        )

    ref_steps = steps_map[ref_dataset_key]
    steps = steps_map[dataset_key]
    if ref_steps <= 0 or steps <= 0:
        return ref_epochs

    target_total_steps = ref_epochs * ref_steps
    epochs = round(target_total_steps / steps)
    return max(min_epochs, int(epochs))


def main() -> None:
    p = argparse.ArgumentParser(
        description="按与 Zhongnan 相同的总 step 数计算 stage1 epoch"
    )
    p.add_argument("--dataset_key", type=str, default="")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--cache_path", type=str, default=DEFAULT_CACHE_PATH)
    p.add_argument("--ref_dataset_key", type=str, default=REF_DATASET_KEY)
    p.add_argument("--ref_epochs", type=int, default=REF_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--refresh_cache", action="store_true")
    p.add_argument("--print_all", action="store_true")
    args = p.parse_args()

    if args.refresh_cache:
        build_steps_cache(
            data_dir=args.data_dir,
            cache_path=args.cache_path,
            batch_size=args.batch_size,
        )

    steps_map = load_steps_cache(args.cache_path)

    if args.print_all:
        ref_steps = steps_map.get(args.ref_dataset_key, 0)
        target = args.ref_epochs * ref_steps
        print(
            f"基准: {args.ref_dataset_key} steps/epoch={ref_steps} "
            f"epochs={args.ref_epochs} -> total_steps={target}\n"
        )
        for key in sorted(steps_map):
            spe = steps_map[key]
            ep = compute_epochs(
                key,
                ref_dataset_key=args.ref_dataset_key,
                ref_epochs=args.ref_epochs,
                steps_map=steps_map,
                batch_size=args.batch_size,
            )
            total = ep * spe
            print(
                f"{key}: steps/epoch={spe} epochs={ep} "
                f"total_steps~{total}"
            )
        return

    if not args.dataset_key:
        p.error("请指定 --dataset_key，或使用 --print_all / --refresh_cache")

    print(
        compute_epochs(
            args.dataset_key,
            ref_dataset_key=args.ref_dataset_key,
            ref_epochs=args.ref_epochs,
            steps_map=steps_map,
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
