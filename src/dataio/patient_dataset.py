from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataio.mae_dataset import _marker_values_to_float32, _read_csv_with_fallback
from dataio.marker_mapping import MARKER_LIST, MARKER_TO_INDEX
from dataio.paths import remap_manifest_path


@dataclass(frozen=True)
class TubeRecord:
    patient_id: str
    csv_path: str
    label: str
    tube_type: str


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    label: str
    tubes: tuple[TubeRecord, ...]


def read_split_rows(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    required = {"patient_id", "csv_path", "label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{xlsx_path}:{sheet_name} missing columns: {missing}")
    out = df.loc[:, [c for c in df.columns if c in {"patient_id", "csv_path", "label", "tube_type"}]].copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["csv_path"] = out["csv_path"].astype(str).map(remap_manifest_path)
    out["label"] = out["label"].astype(str)
    if "tube_type" not in out.columns:
        out["tube_type"] = out["csv_path"].map(infer_tube_type)
    else:
        out["tube_type"] = out["tube_type"].astype(str)
    return out


def infer_tube_type(csv_path: str) -> str:
    """Best-effort tube identity for manifests without an explicit tube_type column."""
    parent = os.path.basename(os.path.dirname(csv_path))
    stem = os.path.basename(csv_path)
    return f"{parent}/{stem}"


def build_label_maps(labels: Iterable[str]) -> tuple[dict[str, int], dict[int, str]]:
    classes = sorted({str(x) for x in labels})
    label_to_idx = {label: i for i, label in enumerate(classes)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    return label_to_idx, idx_to_label


def build_patient_records(
    df: pd.DataFrame,
    *,
    filter_mode: str = "count",
) -> list[PatientRecord]:
    if filter_mode not in {"none", "count", "signature"}:
        raise ValueError("filter_mode must be one of: none, count, signature")

    records: list[PatientRecord] = []
    for patient_id, g in df.groupby("patient_id", sort=False):
        labels = sorted(set(g["label"].astype(str)))
        if len(labels) != 1:
            raise ValueError(f"patient {patient_id} has multiple labels: {labels}")
        tubes = tuple(
            TubeRecord(
                patient_id=str(patient_id),
                csv_path=str(row.csv_path),
                label=str(row.label),
                tube_type=str(row.tube_type),
            )
            for row in g.itertuples(index=False)
        )
        records.append(PatientRecord(str(patient_id), labels[0], tubes))

    if filter_mode == "none" or not records:
        return records

    if filter_mode == "count":
        counts = [len(r.tubes) for r in records]
        keep_count = max(sorted(set(counts)), key=counts.count)
        return [r for r in records if len(r.tubes) == keep_count]

    signatures = [tuple(t.tube_type for t in r.tubes) for r in records]
    keep_signature = max(sorted(set(signatures)), key=signatures.count)
    return [r for r in records if tuple(t.tube_type for t in r.tubes) == keep_signature]


class PatientLevelDataset(Dataset):
    def __init__(
        self,
        xlsx_path: str,
        sheet_name: str,
        *,
        label_to_idx: dict[str, int] | None = None,
        filter_mode: str = "count",
        max_cells_per_tube: int = 5000,
        random_sample: bool = True,
        seed: int = 0,
    ) -> None:
        self.xlsx_path = xlsx_path
        self.sheet_name = sheet_name
        # max_cells_per_tube=None is used by historical evaluations to mean
        # "do not truncate/subsample cells". Internally 0 disables truncation.
        self.max_cells_per_tube = 0 if max_cells_per_tube is None else int(max_cells_per_tube)
        self.random_sample = bool(random_sample)
        self.seed = 0 if seed is None else int(seed)
        self.marker_to_idx = MARKER_TO_INDEX

        df = read_split_rows(xlsx_path, sheet_name)
        self.records = build_patient_records(df, filter_mode=filter_mode)
        if not self.records:
            raise ValueError(f"no patients left after filter_mode={filter_mode}")

        if label_to_idx is None:
            label_to_idx, _ = build_label_maps(r.label for r in self.records)
        self.label_to_idx = label_to_idx
        self.idx_to_label = {i: label for label, i in label_to_idx.items()}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        tube_items = []
        for tube_idx, tube in enumerate(record.tubes):
            values, marker_ids, marker_names, row_ids = self._load_tube(tube.csv_path, idx, tube_idx)
            tube_items.append(
                {
                    "csv_path": tube.csv_path,
                    "tube_type": tube.tube_type,
                    "values": values,
                    "marker_ids": marker_ids,
                    "marker_names": marker_names,
                    "row_ids": row_ids,
                }
            )
        return {
            "patient_id": record.patient_id,
            "label": self.label_to_idx[record.label],
            "label_name": record.label,
            "tubes": tube_items,
        }

    def _load_tube(
        self,
        csv_path: str,
        patient_index: int,
        tube_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
        header = _read_csv_with_fallback(csv_path, nrows=0)
        marker_names = [c for c in header.columns if c in self.marker_to_idx]
        if not marker_names:
            raise ValueError(f"no known marker columns in {csv_path}")

        df = _read_csv_with_fallback(csv_path, usecols=marker_names)
        if self.max_cells_per_tube > 0 and len(df) > self.max_cells_per_tube:
            rng = random.Random(self.seed + patient_index * 1009 + tube_index * 9173)
            rows = sorted(rng.sample(range(len(df)), self.max_cells_per_tube)) if self.random_sample else list(range(self.max_cells_per_tube))
            df = df.iloc[rows]
            row_ids = torch.tensor(rows, dtype=torch.long)
        else:
            row_ids = torch.arange(len(df), dtype=torch.long)

        values_np = _marker_values_to_float32(df)
        values = torch.from_numpy(np.ascontiguousarray(values_np))
        marker_ids = torch.tensor([self.marker_to_idx[c] for c in marker_names], dtype=torch.long)
        return values, marker_ids, marker_names, row_ids


def patient_collate(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError("patient_collate expects batch_size=1 because patient bags are variable length")
    return batch[0]

