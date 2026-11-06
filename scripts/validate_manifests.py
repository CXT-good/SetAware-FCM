"""Validate prepared event tables and patient-disjoint split manifests without training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataio.class_lists import dataset_key_from_split_xlsx
from dataio.marker_mapping import MARKER_TO_INDEX
from dataio.paths import remap_manifest_path


def _read_splits(manifest, splits):
    if manifest.suffix.lower() == ".csv":
        frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        if "split" not in frame:
            raise ValueError("CSV manifests require a split column")
        unknown = set(frame["split"]) - {"train", "validation", "test"}
        if unknown:
            raise ValueError(f"unknown split names: {sorted(unknown)}")
        frames = {name: frame.loc[frame["split"] == name].copy() for name in splits}
    elif manifest.suffix.lower() == ".xlsx":
        with pd.ExcelFile(manifest) as workbook:
            missing = set(splits) - set(workbook.sheet_names)
            if missing:
                raise ValueError(f"missing required split sheets: {sorted(missing)}")
            frames = {name: pd.read_excel(workbook, sheet_name=name, dtype=str, keep_default_na=False) for name in splits}
    else:
        raise ValueError("manifest must be an XLSX workbook or CSV with a split column")
    for name, frame in frames.items():
        if frame.empty:
            raise ValueError(f"missing or empty split: {name}")
    return frames


def _load_classes(class_mapping, dataset_key, manifest):
    if class_mapping is None:
        raise ValueError("class_mapping is required for cell-level validation")
    payload = json.loads(Path(class_mapping).read_text(encoding="utf-8"))
    key = dataset_key or dataset_key_from_split_xlsx(str(manifest))
    if manifest.suffix.lower() == ".csv" and dataset_key is None:
        key = dataset_key_from_split_xlsx(str(manifest.with_suffix(".xlsx")))
    dataset = payload.get("datasets", {}).get(key)
    if dataset is None:
        raise ValueError(f"class mapping has no dataset key: {key}")
    classes = dataset.get("class_list", [])
    counts = dataset.get("class_counts", [])
    if (
        not classes
        or any(not isinstance(label, str) or not label.strip() for label in classes)
        or len(set(classes)) != len(classes)
        or len(counts) != len(classes)
        or any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts)
    ):
        raise ValueError("class mapping needs unique nonempty string class_list and aligned nonnegative integer class_counts")
    return set(classes)


def _event_header(path):
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"):
        try:
            with path.open(encoding=encoding, newline="") as handle:
                columns = next(csv.reader(handle), [])
            if len(columns) != len(set(columns)):
                raise ValueError(f"duplicate event columns: {path.name}")
            return columns, encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode event CSV: {path.name}")


def _validate_events(path, *, level, classes, label_column, strict_marker_columns, ignored_columns, chunksize):
    columns, encoding = _event_header(path)
    ignored = set(ignored_columns) | {label_column, "Time", "event_ID"}
    markers = [column for column in columns if column in MARKER_TO_INDEX and column not in ignored]
    unknown = sorted(set(columns) - set(markers) - ignored)
    if strict_marker_columns and unknown:
        raise ValueError(f"unknown marker or metadata columns in {path.name}: {unknown}; declare metadata with --ignore-column")
    if not markers:
        raise ValueError(f"no known marker columns: {path.name}")
    if level == "cell" and label_column not in columns:
        raise ValueError(f"missing event label column {label_column}: {path.name}")
    selected = markers + ([label_column] if level == "cell" else [])
    event_count = 0
    for chunk in pd.read_csv(path, encoding=encoding, usecols=selected, dtype=str, keep_default_na=False, chunksize=chunksize):
        values = chunk[markers].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"non-numeric or non-finite marker values: {path.name}, rows {event_count + 1} onward")
        if level == "cell":
            labels = chunk[label_column].str.strip()
            if labels.isin(["", "nan"]).any():
                raise ValueError(f"empty event labels: {path.name}")
            if not set(labels).issubset(classes):
                raise ValueError(f"unmapped event labels: {path.name}; review the configured class mapping")
        event_count += len(chunk)
    if not event_count:
        raise ValueError(f"no events: {path.name}")
    return event_count, unknown


def validate_manifest(
    manifest,
    *,
    level,
    splits=("train", "validation", "test"),
    class_mapping=None,
    dataset_key=None,
    label_column="label",
    strict_marker_columns=False,
    ignored_columns=(),
    chunksize=100000,
):
    if level not in {"cell", "patient"}:
        raise ValueError("level must be cell or patient")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or "train" not in splits or set(splits) - {"train", "validation", "test"}:
        raise ValueError("splits must be unique train/validation/test names and include train")
    manifest = Path(manifest).resolve()
    frames = _read_splits(manifest, splits)
    classes = _load_classes(class_mapping, dataset_key, manifest) if level == "cell" else None
    seen_paths = {}
    seen_patients = {}
    patient_labels = {}
    ignored_event_columns = set()
    result = {"status": "passed", "level": level, "patient_disjointness": "checked", "splits": {}}
    train_labels = set(frames["train"]["label"].astype(str).str.strip()) if "label" in frames["train"] else set()
    for split, frame in frames.items():
        required = {"patient_id", "csv_path"} | ({"label"} if level == "patient" else set())
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{split}: missing columns {sorted(missing)}; patient_id is required to check leakage")
        for column in required:
            frame[column] = frame[column].astype(str).str.strip()
            if frame[column].eq("").any():
                raise ValueError(f"{split}: empty required {column}")
        event_count = 0
        for row in frame.to_dict("records"):
            patient_id = row["patient_id"]
            path = Path(remap_manifest_path(row["csv_path"])).resolve()
            if path in seen_paths:
                if seen_paths[path] != split:
                    raise ValueError(f"CSV appears across splits: {path.name}")
                raise ValueError(f"{split}: duplicate CSV tube row: {path.name}")
            if patient_id in seen_patients and seen_patients[patient_id] != split:
                raise ValueError("patient appears across splits; keep all cells and tubes from each patient together")
            seen_paths[path] = split
            seen_patients[patient_id] = split
            if not path.is_file():
                raise ValueError(f"{split}: event CSV does not exist: {path}; relative paths use the process working directory")
            if path.suffix != ".csv":
                raise ValueError(f"event path must end in .csv: {path.name}")
            if level == "patient":
                label = row["label"]
                if patient_id in patient_labels and patient_labels[patient_id] != label:
                    raise ValueError("inconsistent patient labels across tube rows")
                patient_labels[patient_id] = label
                if split != "train" and label not in train_labels:
                    raise ValueError(f"{split}: patient label is not in train; review the patient-level label mapping")
            events, unknown = _validate_events(
                path, level=level, classes=classes, label_column=label_column,
                strict_marker_columns=strict_marker_columns, ignored_columns=ignored_columns, chunksize=chunksize,
            )
            event_count += events
            ignored_event_columns.update(unknown)
        result["splits"][split] = {"patients": int(frame.patient_id.nunique()), "tubes": len(frame), "events": event_count}
    result["ignored_event_columns"] = sorted(ignored_event_columns)
    result["scope"] = "Prepared CSV contract and selected splits only; does not establish historical cohort identity or validate external FCS preprocessing."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--level", choices=("cell", "patient"), required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--class-mapping", type=Path)
    parser.add_argument("--dataset-key")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--strict-marker-columns", action="store_true")
    parser.add_argument("--ignore-column", action="append", default=[])
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = validate_manifest(
            args.manifest, level=args.level, splits=args.splits, class_mapping=args.class_mapping,
            dataset_key=args.dataset_key, label_column=args.label_column,
            strict_marker_columns=args.strict_marker_columns, ignored_columns=args.ignore_column, chunksize=args.chunksize,
        )
    except (ValueError, OSError, KeyError, pd.errors.ParserError) as error:
        parser.exit(2, f"Manifest validation failed: {error}\n")
    serialized = json.dumps(result, indent=2)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
