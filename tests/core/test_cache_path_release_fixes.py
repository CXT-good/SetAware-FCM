import json
import os

import numpy as np
import pandas as pd
import pytest

from cell_level.baselines.set_transformer.helpers import build_uniform_tube_cache, read_split_paths
from cell_level.ctransformer.eval import _collect_test_csv_paths


def _cache(csv_path, cache_dir, **options):
    config = {
        "marker_columns": ["CD10", "CD19"],
        "class_to_index": {"negative": 0, "positive": 1},
        "max_cells": 8,
        "seed": 13,
        **options,
    }
    return build_uniform_tube_cache(str(csv_path), cache_dir=str(cache_dir), **config)


def test_cache_invalidates_source_changes_even_with_same_size_and_mtime(tmp_path):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("CD10,CD19,label\n1,3,negative\n2,4,positive\n", encoding="utf-8")
    source_stat = csv_path.stat()
    _cache(csv_path, tmp_path / "cache")

    csv_path.write_text("CD10,CD19,label\n9,7,positive\n8,6,negative\n", encoding="utf-8")
    assert csv_path.stat().st_size == source_stat.st_size
    os.utime(csv_path, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
    x_path, y_path, rows = _cache(csv_path, tmp_path / "cache")

    assert rows == 2
    np.testing.assert_array_equal(np.load(x_path), [[9, 7], [8, 6]])
    np.testing.assert_array_equal(np.load(y_path), [1, 0])


@pytest.mark.parametrize("damage", ["short_labels", "wrong_label_dtype", "unreadable_labels"])
def test_cache_rebuilds_invalid_label_array(tmp_path, damage):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("CD10,CD19,label\n1,3,negative\n2,4,positive\n", encoding="utf-8")
    _, y_path, _ = _cache(csv_path, tmp_path / "cache")
    if damage == "unreadable_labels":
        with open(y_path, "wb") as handle:
            handle.write(b"incomplete cache")
    else:
        values = np.array([0], dtype=np.int64) if damage == "short_labels" else np.array([0, 1], dtype=np.float32)
        np.save(y_path, values)

    x_path, y_path, rows = _cache(csv_path, tmp_path / "cache")

    assert rows == 2
    assert np.load(y_path).dtype == np.int64
    np.testing.assert_array_equal(np.load(y_path), [0, 1])
    np.testing.assert_array_equal(np.load(x_path), [[1, 3], [2, 4]])


def test_cache_reuses_equivalent_class_mapping_without_rewriting_arrays(tmp_path):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("CD10,CD19,label\n1,3,negative\n2,4,positive\n", encoding="utf-8")
    initial = _cache(csv_path, tmp_path / "cache")
    for path in initial[:2]:
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))

    repeated = _cache(
        csv_path, tmp_path / "cache", class_to_index={"positive": 1, "negative": 0}
    )

    assert repeated == initial
    assert all(os.stat(path).st_mtime_ns == 1_000_000_000 for path in repeated[:2])


@pytest.mark.parametrize("reader", ["ctransformer", "set_transformer"])
def test_manifest_readers_use_specific_prefix_and_preserve_unmapped_paths(tmp_path, monkeypatch, reader):
    general = tmp_path / "general"
    specific = tmp_path / "specific"
    general.mkdir()
    specific.mkdir()
    paths = [specific / "nested.csv", general / "other.csv", tmp_path / "local.csv"]
    for path in paths:
        path.write_text("CD19,label\n1,negative\n", encoding="utf-8")
    archived = str(tmp_path / "archive")
    workbook = tmp_path / "split.xlsx"
    pd.DataFrame({"csv_path": [
        " " + archived + "/nested/nested.csv ", archived + "/other.csv", str(paths[2]),
        archived + "/missing.csv", None,
    ]}).to_excel(workbook, sheet_name="test", index=False)
    monkeypatch.setenv("FCM_ML_PATH_MAP", json.dumps({
        archived: str(general), archived + "/nested": str(specific),
    }))

    if reader == "ctransformer":
        observed = _collect_test_csv_paths(str(workbook), csv_col="csv_path")
    else:
        observed = read_split_paths(str(workbook), "test")

    assert observed == [str(path) for path in paths]


@pytest.mark.parametrize("reader", ["ctransformer", "set_transformer"])
def test_manifest_readers_reject_invalid_path_map(tmp_path, monkeypatch, reader):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("CD19,label\n1,negative\n", encoding="utf-8")
    workbook = tmp_path / "split.xlsx"
    pd.DataFrame({"csv_path": [str(csv_path)]}).to_excel(workbook, sheet_name="test", index=False)
    monkeypatch.setenv("FCM_ML_PATH_MAP", "[]")

    with pytest.raises(ValueError, match="FCM_ML_PATH_MAP"):
        if reader == "ctransformer":
            _collect_test_csv_paths(str(workbook), csv_col="csv_path")
        else:
            read_split_paths(str(workbook), "test")
