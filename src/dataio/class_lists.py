from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DatasetClassInfo:
    dataset_key: str
    class_list: List[str]
    class_counts: List[int]

    @property
    def class_to_index(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.class_list)}


# overall 行 macro P/R/F1 计算时排除的类别名（按数据集）
MACRO_EXCLUDE_CLASS_NAMES: Dict[str, List[str]] = {}
# 测试集 support=0 的类别也不参与 macro（按数据集）
MACRO_EXCLUDE_ZERO_SUPPORT_DATASETS: set[str] = {"Zhongnan_Hospital"}


def macro_label_indices(
    class_list: List[str],
    dataset_key: Optional[str] = None,
    *,
    supports: Optional[List[int]] = None,
) -> List[int]:
    """返回参与 macro P/R/F1 的类别索引。"""
    exclude = set(MACRO_EXCLUDE_CLASS_NAMES.get(dataset_key or "", []))
    exclude_zero = (dataset_key or "") in MACRO_EXCLUDE_ZERO_SUPPORT_DATASETS
    out: List[int] = []
    for i, name in enumerate(class_list):
        if name in exclude:
            continue
        if exclude_zero and supports is not None and int(supports[i]) == 0:
            continue
        out.append(i)
    return out


def dataset_key_from_split_xlsx(xlsx_path: str) -> str:
    """从 *_dataset_split.xlsx 推导稳定的数据集 key。"""
    base = os.path.basename(xlsx_path)
    for suf in (".xlsx", ".xls"):
        if base.lower().endswith(suf):
            base = base[: -len(suf)]
    if base.lower().endswith("_dataset_split"):
        base = base[: -len("_dataset_split")]
    # fold 划分文件（如 *_dataset_split_fold4.xlsx）应复用原数据集 key
    # 避免 class mapping 中出现重复 key。
    base = re.sub(r"_dataset_split_fold\d+$", "", base, flags=re.IGNORECASE)
    return base


def save_class_mappings_json(
    mapping_path: str,
    *,
    datasets: Dict[str, DatasetClassInfo],
) -> None:
    payload = {
        "version": 1,
        "datasets": {
            k: {
                "class_list": v.class_list,
                "class_counts": v.class_counts,
            }
            for k, v in sorted(datasets.items())
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(mapping_path)), exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_class_mappings_json(mapping_path: str) -> Dict[str, DatasetClassInfo]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    datasets_raw = payload.get("datasets", {})
    out: Dict[str, DatasetClassInfo] = {}
    for k, v in datasets_raw.items():
        cls = list(v.get("class_list", []))
        counts = list(v.get("class_counts", []))
        out[k] = DatasetClassInfo(dataset_key=k, class_list=cls, class_counts=counts)
    return out


def load_dataset_class_info(
    mapping_path: str,
    *,
    dataset_key: str,
) -> DatasetClassInfo:
    mappings = load_class_mappings_json(mapping_path)
    if dataset_key not in mappings:
        raise KeyError(
            f"class mapping 未找到 dataset_key={dataset_key}。"
            f"请先生成映射文件：{mapping_path}"
        )
    return mappings[dataset_key]

