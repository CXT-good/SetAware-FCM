"""
预先统计所有数据集的逐细胞类别（CLASS_LIST / CLASS_NUMBER），并保存为 JSON。

默认会遍历 data_dir 下的 *_dataset_split.xlsx，并读取每个 xlsx 的 train sheet，
再对 train 的所有 CSV 扫描标签列（默认 label），输出每个数据集自己的 class_list 与 class_counts。
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import pandas as pd

from dataio.class_lists import (
    DatasetClassInfo,
    dataset_key_from_split_xlsx,
    load_class_mappings_json,
    save_class_mappings_json,
)


def _list_split_xlsx(data_dir: str) -> list[str]:
    out: list[str] = []
    for f in sorted(os.listdir(data_dir)):
        if f.lower().endswith("_dataset_split.xlsx"):
            out.append(os.path.join(data_dir, f))
    return out


def _read_sheet_csv_paths(xlsx_path: str, *, sheet: str, csv_col: str) -> list[str]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    if csv_col not in df.columns:
        raise ValueError(
            f"{xlsx_path} sheet={sheet} 缺少列 {csv_col}，当前列: {list(df.columns)}"
        )
    out: list[str] = []
    for raw in df[csv_col].astype(str).tolist():
        p = os.path.abspath(raw.strip())
        if not p.lower().endswith(".csv"):
            continue
        if os.path.isfile(p):
            out.append(p)
    return out


def _read_label_series(csv_path: str, *, label_col: str) -> list[str]:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"):
        try:
            s = pd.read_csv(
                csv_path,
                usecols=[label_col],
                encoding=encoding,
                dtype={label_col: str},
            )[label_col]
            return [str(v).strip() if pd.notna(v) and str(v).strip().lower() != "nan" else "" for v in s.tolist()]
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"无法读取标签列: {csv_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="构建所有数据集的 CLASS_LIST/CLASS_NUMBER 映射（JSON）")
    p.add_argument("--data_dir", type=str, default="data", help="包含 split xlsx 的目录")
    p.add_argument(
        "--output_path",
        type=str,
        default="class_mappings/datasets_class_mappings.json",
        help="输出 JSON 路径",
    )
    p.add_argument(
        "--sheet",
        type=str,
        default="total",
        help="从哪个子表读取 csv_path（默认 total；也可用 train/test）",
    )
    p.add_argument("--xlsx_csv_col", type=str, default="csv_path")
    p.add_argument("--csv_label_col", type=str, default="label")
    p.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="每个数据集最多扫描多少个 train CSV（0 表示不限制）",
    )
    p.add_argument(
        "--progress_every",
        type=int,
        default=5,
        help="每扫描多少个 CSV 打印一次进度（默认 5）",
    )
    p.add_argument(
        "--dataset_key",
        type=str,
        default="",
        help="仅处理指定数据集 key（默认处理全部）",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="与已有 JSON 合并（仅更新本次扫描到的数据集）",
    )
    args = p.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    xlsx_files = _list_split_xlsx(data_dir)
    if not xlsx_files:
        raise ValueError(f"在 {data_dir} 下未找到 *_dataset_split.xlsx")

    datasets: dict[str, DatasetClassInfo] = {}
    if args.merge and os.path.isfile(args.output_path):
        datasets = load_class_mappings_json(args.output_path)
    skipped = 0

    print(f"发现 split xlsx 数量: {len(xlsx_files)}")
    for xlsx_path in xlsx_files:
        key = dataset_key_from_split_xlsx(xlsx_path)
        if args.dataset_key and key != args.dataset_key:
            continue
        try:
            csvs = _read_sheet_csv_paths(
                xlsx_path, sheet=args.sheet, csv_col=args.xlsx_csv_col
            )
            if not csvs:
                print(f"[跳过] {key}: sheet={args.sheet} 为空")
                skipped += 1
                continue
            if args.max_files:
                csvs = csvs[: args.max_files]

            counter: Counter[str] = Counter()
            total_files = len(csvs)
            print(f"[scan] {key}: files={total_files} sheet={args.sheet}")
            for i, csv_path in enumerate(csvs, start=1):
                labs = _read_label_series(csv_path, label_col=args.csv_label_col)
                counter.update([x for x in labs if x != ""])
                if args.progress_every and (i % args.progress_every == 0 or i == total_files):
                    print(f"  - {key}: {i}/{total_files} 已扫描")

            class_list = [c for c, _n in counter.most_common()]
            class_counts = [int(counter[c]) for c in class_list]
            datasets[key] = DatasetClassInfo(
                dataset_key=key, class_list=class_list, class_counts=class_counts
            )
            print(f"[ok] {key}: classes={len(class_list)} files={total_files}")
        except Exception as e:
            print(f"[跳过] {key}: {e}")
            skipped += 1

    if not datasets:
        raise ValueError("没有成功生成任何数据集的 class mapping")

    save_class_mappings_json(args.output_path, datasets=datasets)
    print(f"\n已保存: {os.path.abspath(args.output_path)}")
    print(f"成功: {len(datasets)}  跳过: {skipped}")


if __name__ == "__main__":
    main()

