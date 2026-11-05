from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import os
import subprocess
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = ROOT / "scripts" / "validate_manifests.py"
    assert path.is_file(), "The release needs a manifest validator before training"
    spec = spec_from_file_location("validate_manifests", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_manifest(tmp_path, *, level="patient", csv=False):
    frames = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}_events.csv"
        pd.DataFrame({"CD3": [0.1, 0.2], "CD45": [1.0, 2.0], "label": ["0", "1"]}).to_csv(path, index=False)
        frames[split] = pd.DataFrame({
            "patient_id": [f"synthetic_{split}"], "csv_path": [str(path)],
            "label": ["class_0"], "tube_type": ["panel_0"],
        })
    manifest = tmp_path / ("synthetic_dataset_split.csv" if csv else "synthetic_dataset_split.xlsx")
    write_manifest(manifest, frames)
    mapping = tmp_path / "classes.json"
    mapping.write_text(json.dumps({"version": 1, "datasets": {"synthetic": {
        "class_list": ["0", "1"], "class_counts": [1, 1],
    }}}), encoding="utf-8")
    options = {"level": level}
    if level == "cell":
        options["class_mapping"] = mapping
    return manifest, frames, options


def write_manifest(manifest, frames):
    if manifest.suffix == ".csv":
        pd.concat([frame.assign(split=split) for split, frame in frames.items()]).to_csv(manifest, index=False)
    else:
        with pd.ExcelWriter(manifest, engine="openpyxl") as writer:
            for split, frame in frames.items():
                frame.to_excel(writer, sheet_name=split, index=False)


@pytest.mark.parametrize("level", ["cell", "patient"])
@pytest.mark.parametrize("csv", [False, True])
def test_validates_artificial_disjoint_manifests(tmp_path, level, csv):
    manifest, frames, options = make_manifest(tmp_path, level=level, csv=csv)
    result = load_validator().validate_manifest(manifest, **options)
    assert result["status"] == "passed"
    assert result["splits"]["test"] == {"patients": 1, "tubes": 1, "events": 2}


def test_missing_split_never_falls_back_to_train(tmp_path):
    manifest, frames, options = make_manifest(tmp_path)
    frames.pop("test")
    write_manifest(manifest, frames)
    with pytest.raises(ValueError, match="missing.*test"):
        load_validator().validate_manifest(manifest, **options)


def test_two_partition_protocol_requires_explicit_selection(tmp_path):
    manifest, frames, options = make_manifest(tmp_path)
    frames.pop("validation")
    write_manifest(manifest, frames)
    result = load_validator().validate_manifest(manifest, splits=("train", "test"), **options)
    assert set(result["splits"]) == {"train", "test"}


@pytest.mark.parametrize("field,value,match", [
    ("patient_id", "synthetic_train", "patient.*across splits"),
    ("csv_path", "train", "CSV.*across splits"),
    ("csv_path", "absent.csv", "does not exist"),
    ("label", "unseen", "patient label.*not in train"),
    ("patient_id", "", "empty.*patient_id"),
])
def test_rejects_unsafe_manifests(tmp_path, field, value, match):
    manifest, frames, options = make_manifest(tmp_path)
    if value == "train":
        value = frames["train"].at[0, "csv_path"]
    frames["test"].at[0, field] = value
    write_manifest(manifest, frames)
    with pytest.raises(ValueError, match=match):
        load_validator().validate_manifest(manifest, **options)


def test_rejects_duplicate_tube_rows(tmp_path):
    manifest, frames, options = make_manifest(tmp_path)
    frames["train"] = pd.concat([frames["train"], frames["train"]], ignore_index=True)
    write_manifest(manifest, frames)
    with pytest.raises(ValueError, match="duplicate CSV"):
        load_validator().validate_manifest(manifest, **options)


def test_rejects_inconsistent_patient_labels(tmp_path):
    manifest, frames, options = make_manifest(tmp_path)
    extra = frames["train"].copy()
    extra["csv_path"] = str(tmp_path / "extra.csv")
    extra["label"] = "class_1"
    pd.read_csv(frames["train"].at[0, "csv_path"]).to_csv(extra.at[0, "csv_path"], index=False)
    frames["train"] = pd.concat([frames["train"], extra], ignore_index=True)
    write_manifest(manifest, frames)
    with pytest.raises(ValueError, match="inconsistent patient labels"):
        load_validator().validate_manifest(manifest, **options)


@pytest.mark.parametrize("contents,match", [
    ("CD3,unknown_marker,label\n1,2,0\n", "unknown marker"),
    ("CD3,label\ninvalid,0\n", "non-numeric or non-finite"),
    ("CD3,label\ninf,0\n", "non-numeric or non-finite"),
    ("CD3,label\n1,not_mapped\n", "unmapped event labels"),
    ("CD3,label\n1,\n", "empty event labels"),
    ("CD3\n1\n", "missing event label"),
    ("CD3,label\n", "no events"),
    ("CD3,CD3,label\n1,2,0\n", "duplicate event columns"),
])
def test_cell_event_checks_prevent_silent_coercion(tmp_path, contents, match):
    manifest, frames, options = make_manifest(tmp_path, level="cell")
    Path(frames["train"].at[0, "csv_path"]).write_text(contents, encoding="utf-8")
    options["strict_marker_columns"] = True
    with pytest.raises(ValueError, match=match):
        load_validator().validate_manifest(manifest, **options)


def test_ignored_metadata_is_reported_not_treated_as_marker(tmp_path):
    manifest, frames, options = make_manifest(tmp_path)
    path = Path(frames["train"].at[0, "csv_path"])
    frame = pd.read_csv(path)
    frame["Time"] = [0, 1]
    frame["operator_note"] = ["artificial", "artificial"]
    frame.to_csv(path, index=False)
    result = load_validator().validate_manifest(manifest, **options)
    assert result["ignored_event_columns"] == ["operator_note"]


def test_relative_paths_follow_loader_working_directory(tmp_path, monkeypatch):
    manifest, frames, options = make_manifest(tmp_path)
    for frame in frames.values():
        frame["csv_path"] = frame["csv_path"].map(lambda path: Path(path).name)
    write_manifest(manifest, frames)
    monkeypatch.chdir(tmp_path)
    assert load_validator().validate_manifest(manifest, **options)["status"] == "passed"
    monkeypatch.chdir(tmp_path.parent)
    with pytest.raises(ValueError, match="does not exist"):
        load_validator().validate_manifest(manifest, **options)


def test_explicit_path_prefix_mapping_matches_loaders(tmp_path, monkeypatch):
    manifest, frames, options = make_manifest(tmp_path)
    for frame in frames.values():
        frame["csv_path"] = frame["csv_path"].map(lambda path: "/archive/events/" + Path(path).name)
    write_manifest(manifest, frames)
    monkeypatch.setenv("FCM_ML_PATH_MAP", json.dumps({"/archive/events": str(tmp_path)}))
    assert load_validator().validate_manifest(manifest, **options)["status"] == "passed"


def test_cell_validation_requires_explicit_class_mapping(tmp_path):
    manifest, frames, options = make_manifest(tmp_path, level="cell")
    options.pop("class_mapping")
    with pytest.raises(ValueError, match="class.mapping is required"):
        load_validator().validate_manifest(manifest, **options)


def test_csv_manifest_rejects_unknown_partition_names(tmp_path):
    manifest, frames, options = make_manifest(tmp_path, csv=True)
    frame = pd.read_csv(manifest)
    frame.loc[frame["split"] == "test", "split"] = "tset"
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="unknown split"):
        load_validator().validate_manifest(manifest, **options)


def test_utf8_bom_does_not_change_marker_identity(tmp_path):
    manifest, frames, options = make_manifest(tmp_path)
    path = Path(frames["train"].at[0, "csv_path"])
    path.write_text("\ufeffCD3,label\n1,0\n", encoding="utf-8")
    assert load_validator().validate_manifest(manifest, **options)["status"] == "passed"


def test_cli_validation_failure_returns_nonzero_without_traceback(tmp_path):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_manifests.py"),
         "--manifest", str(tmp_path / "absent.xlsx"), "--level", "patient"],
        env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "Manifest validation failed" in result.stderr
    assert "Traceback" not in result.stderr
