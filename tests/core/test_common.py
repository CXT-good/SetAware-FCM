from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cell_level.multicenter.common import (
    COMMON_MARKERS,
    iter_csv_chunks,
    load_source_training_data,
    write_metric_artifacts,
)


def _make_csv(tmp_path, *, drop: str | None = None, label: str = "A"):
    columns = {marker: [float(i), float(i + 1)] for i, marker in enumerate(COMMON_MARKERS)}
    if drop is not None:
        columns.pop(drop)
    columns["label"] = [label, label]
    path = tmp_path / "tube.csv"
    pd.DataFrame(columns).to_csv(path, index=False)
    return path


def test_common_marker_order_is_fixed():
    assert COMMON_MARKERS == (
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


def test_missing_marker_raises(tmp_path):
    path = _make_csv(tmp_path, drop="CD10")
    with pytest.raises(ValueError, match="missing required markers"):
        list(iter_csv_chunks(str(path), COMMON_MARKERS, {"A": 0}))


def test_unknown_label_raises(tmp_path):
    path = _make_csv(tmp_path, label="B")
    with pytest.raises(ValueError, match="unknown labels"):
        list(iter_csv_chunks(str(path), COMMON_MARKERS, {"A": 0}))


def test_metrics_and_four_decimal_export(tmp_path):
    cm = np.array([[8, 2], [1, 9]], dtype=np.int64)
    write_metric_artifacts(cm, ["A", "B"], tmp_path, {"method": "unit"})

    overall_text = (tmp_path / "overall_metrics.csv").read_text(encoding="utf-8")
    per_class_text = (tmp_path / "metrics_per_class.csv").read_text(encoding="utf-8")
    meta = json.loads((tmp_path / "eval_meta.json").read_text(encoding="utf-8"))
    assert "0.8500" in overall_text
    assert "0.8000" in per_class_text
    assert meta["method"] == "unit"
    np.testing.assert_array_equal(np.load(tmp_path / "agg_confusion.npy"), cm)


def test_source_training_loader_uses_fixed_order_and_cap(tmp_path):
    csv_path = _make_csv(tmp_path)
    split_path = tmp_path / "center_dataset_split.xlsx"
    with pd.ExcelWriter(split_path) as writer:
        pd.DataFrame({"csv_path": [str(csv_path)]}).to_excel(
            writer, sheet_name="train", index=False
        )
    x, y, paths = load_source_training_data(
        str(split_path), ["A"], max_cells_per_file=1, seed=42
    )
    assert x.shape == (1, len(COMMON_MARKERS))
    assert y.tolist() == [0]
    assert paths == [str(csv_path.resolve())]
