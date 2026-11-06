import json

import pandas as pd
import pytest

from dataio.baseline_loaders import collect_csv_paths_from_xlsx
from dataio.paths import remap_manifest_path
from common.patient_baselines import read_split_book
from dataio.patient_dataset import read_split_rows
from cell_level.multicenter.common import read_split_paths
from cell_level.ctransformer.compute_epochs import _train_csv_paths


def test_prefix_mapping_preserves_unmapped_paths_and_uses_path_boundaries(monkeypatch, tmp_path):
    monkeypatch.delenv("FCM_ML_PATH_MAP", raising=False)
    assert remap_manifest_path("relative/tube.csv") == "relative/tube.csv"
    mapping = {"/archive/data": str(tmp_path / "fallback"), "/archive/data/nested": str(tmp_path / "specific")}
    monkeypatch.setenv("FCM_ML_PATH_MAP", json.dumps(mapping))
    assert remap_manifest_path("/archive/data/nested/tube.csv") == str(tmp_path / "specific/tube.csv")
    assert remap_manifest_path("/archive/databases/tube.csv") == "/archive/databases/tube.csv"


def test_split_readers_remap_before_filtering_missing_files(monkeypatch, tmp_path):
    events = tmp_path / "events"
    events.mkdir()
    csv_path = events / "tube.csv"
    pd.DataFrame({"CD3": [1.0], "label": [0]}).to_csv(csv_path, index=False)
    frame = pd.DataFrame({"patient_id": ["synthetic_000"], "csv_path": ["/archive/data/tube.csv"], "label": ["class_0"]})
    workbook = tmp_path / "split.xlsx"
    frame.to_excel(workbook, sheet_name="train", index=False)
    monkeypatch.setenv("FCM_ML_PATH_MAP", json.dumps({"/archive/data": str(events)}))
    assert collect_csv_paths_from_xlsx(str(workbook)) == [str(csv_path)]
    assert read_split_rows(str(workbook), "train").csv_path.tolist() == [str(csv_path)]
    assert read_split_book(str(workbook), "train")._csv_path.tolist() == [str(csv_path)]
    assert read_split_paths(str(workbook), "train") == [str(csv_path)]
    assert _train_csv_paths(str(workbook)) == [str(csv_path)]


def test_invalid_prefix_mapping_reports_configuration_error(monkeypatch):
    monkeypatch.setenv("FCM_ML_PATH_MAP", "[]")
    with pytest.raises(ValueError, match="FCM_ML_PATH_MAP"):
        remap_manifest_path("tube.csv")
