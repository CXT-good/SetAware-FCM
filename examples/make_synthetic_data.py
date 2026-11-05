"""Create artificial event CSVs and disjoint patient-level split workbooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataio.marker_mapping import LABEL_COLUMN, MARKER_LIST


def make_data(destination: Path, seed: int = 42) -> dict:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)
    preferred_markers = ["CD3", "CD4", "CD8", "CD19", "CD34", "CD45"]
    markers = [marker for marker in preferred_markers if marker in MARKER_LIST]
    if len(markers) < 6:
        markers = list(MARKER_LIST[:6])
    splits = {}
    class_counts = {"0": 0, "1": 0}
    patient_counter = 0
    for split_name, patient_count in [("train", 4), ("validation", 2), ("test", 2)]:
        rows = []
        for patient_index in range(patient_count):
            patient_id = f"synthetic_{patient_counter:03d}"
            patient_counter += 1
            for tube_index in range(2):
                values = generator.normal(size=(32, len(markers))).astype(np.float32)
                labels = (values[:, 0] > 0).astype(np.int64)
                frame = pd.DataFrame(values, columns=markers)
                frame[LABEL_COLUMN] = labels
                tube_directory = destination / "events" / split_name
                tube_directory.mkdir(parents=True, exist_ok=True)
                csv_path = tube_directory / f"{patient_id}_tube_{tube_index}.csv"
                frame.to_csv(csv_path, index=False)
                rows.append({
                    "patient_id": patient_id,
                    "csv_path": str(csv_path),
                    "label": f"class_{patient_index % 2}",
                    "tube_type": f"panel_{tube_index}",
                })
                if split_name == "train":
                    for label, count in zip(*np.unique(labels, return_counts=True)):
                        class_counts[str(int(label))] += int(count)
        splits[split_name] = pd.DataFrame(rows)
    patient_book = destination / "patient_split.xlsx"
    cell_book = destination / "synthetic_dataset_split.xlsx"
    for workbook in [patient_book, cell_book]:
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            for split_name, frame in splits.items():
                frame.to_excel(writer, sheet_name=split_name, index=False)
    mapping = {"version": 1, "datasets": {"synthetic": {"class_list": ["0", "1"], "class_counts": [class_counts["0"], class_counts["1"]]}}}
    (destination / "class_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    metadata = {
        "description": "Artificial values and identifiers; no real patient data.",
        "seed": seed,
        "markers": markers,
        "event_label_column": LABEL_COLUMN,
        "patients": {split: int(frame.patient_id.nunique()) for split, frame in splits.items()},
    }
    (destination / "synthetic_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(make_data(args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
