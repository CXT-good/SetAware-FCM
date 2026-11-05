import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from cell_level.baselines import _eval, _train
from cell_level.baselines.logistic_regression.model import train_eval_logistic_regression
from cell_level.baselines.set_transformer.helpers import build_uniform_tube_cache
from cell_level.baselines.set_transformer.helpers import read_split_paths
from cell_level.ctransformer.eval import _collect_test_csv_paths
from cell_level.pactransformer.eval import bootstrap_ci_binary, safe_prauc
from common.patient_baselines import read_split_book
from common.patient_paper import patient_table
from dataio.baseline_loaders import collect_csv_paths_from_xlsx
from patient_level.baselines.gmm_fisher_svm.model_per_tube import load_one_file


def write_cell_split(directory, features, targets, class_list):
    directory.mkdir(parents=True, exist_ok=True)
    markers = ["CD10", "CD19"][: features.shape[1]]
    frame = pd.DataFrame(features, columns=markers)
    frame["label"] = [class_list[index] for index in targets]
    csv_path = directory / "events.csv"
    frame.to_csv(csv_path, index=False)
    workbook = directory / "synthetic_dataset_split.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        for split in ("train", "test"):
            pd.DataFrame({"csv_path": [str(csv_path)]}).to_excel(
                writer, sheet_name=split, index=False
            )
    mapping = directory / "classes.json"
    mapping.write_text(
        json.dumps(
            {
                "version": 1,
                "datasets": {
                    "synthetic": {
                        "class_list": class_list,
                        "class_counts": [int(np.sum(targets == index)) for index in range(len(class_list))],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return csv_path, workbook, mapping


@pytest.mark.parametrize("noncontiguous", [False, True], ids=["binary", "noncontiguous"])
def test_lr_saved_model_matches_training_predictions(tmp_path, monkeypatch, noncontiguous):
    monkeypatch.delenv("FCM_ML_MARKER_COLUMNS", raising=False)
    if noncontiguous:
        features = np.array(
            [[-4, -4], [-4, -3], [4, -4], [4, -3], [0, 4], [0, 3]],
            dtype=np.float32,
        )
        targets = np.array([0, 0, 2, 2, 3, 3])
        class_list = ["class_0", "absent", "class_2", "class_3"]
    else:
        features = np.array([[-3], [-2], [-1], [1], [2], [3]], dtype=np.float32)
        targets = np.array([0, 0, 0, 1, 1, 1])
        class_list = ["negative", "positive"]
    _, workbook, mapping = write_cell_split(tmp_path, features, targets, class_list)
    model_dir = tmp_path / "model"
    train_eval_logistic_regression(
        features, targets, features, targets, str(model_dir), class_list=class_list
    )
    expected = pd.read_csv(model_dir / "confusion_matrix.csv", index_col=0).to_numpy()
    output_dir = tmp_path / "eval"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval", "--model_dir", str(model_dir), "--dataset_xlsx", str(workbook),
            "--class_mapping_path", str(mapping), "--device", "cpu",
            "--output_dir", str(output_dir), "--warmup_batches", "0",
        ],
    )
    _eval.main("lr")
    observed = pd.read_csv(output_dir / "confusion_matrix.csv", index_col=0).to_numpy()
    np.testing.assert_array_equal(observed, expected)


def test_patient_reader_rejects_missing_split(tmp_path):
    csv_path = tmp_path / "events.csv"
    pd.DataFrame({"CD10": [1.0]}).to_csv(csv_path, index=False)
    workbook = tmp_path / "train_only.xlsx"
    pd.DataFrame(
        {"patient_id": ["synthetic_patient"], "csv_path": [str(csv_path)], "label": ["normal"]}
    ).to_excel(workbook, sheet_name="train", index=False)

    with pytest.raises(ValueError):
        read_split_book(str(workbook), "test")


@pytest.fixture(scope="module")
def replayed_patient_features(tmp_path_factory):
    directory = tmp_path_factory.mktemp("patient_replay")
    csv_path = directory / "patient_tube_1.csv"
    values = np.random.default_rng(9).normal(size=(128, 2)).astype(np.float32)
    pd.DataFrame(values, columns=["CD19", "CD20"]).to_csv(csv_path, index=False)
    program = """
import json
import sys
import pandas as pd
from patient_level.baselines.flowcat.model import patient_tensor
from patient_level.baselines.gmm_fisher_svm.model import load_cells_for_patient
from patient_level.baselines.gmm_fisher_svm.model_per_tube import load_one_file
rows = pd.DataFrame({'_patient_id': ['synthetic_patient'], '_csv_path': [sys.argv[1]], '_tube_id': ['1']})
features = {
    'flowcat': patient_tensor(rows, 'synthetic_patient', ['1'], 3, 128, 32, 13).tolist(),
    'gmm_patient': load_cells_for_patient(rows, 'synthetic_patient', ['CD19', 'CD20'], 8, 13).tolist(),
    'gmm_file': load_one_file(sys.argv[1], ['CD19', 'CD20'], 8, 13).tolist(),
}
print(json.dumps(features))
"""
    outputs = []
    source_root = Path(__file__).resolve().parents[2] / "src"
    for hash_seed in (1, 2):
        environment = os.environ.copy()
        environment.update(
            PYTHONHASHSEED=str(hash_seed),
            PYTHONPATH=str(source_root),
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            CUDA_VISIBLE_DEVICES="",
        )
        result = subprocess.run(
            [sys.executable, "-c", program, str(csv_path)],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        outputs.append(json.loads(result.stdout))
    return outputs


@pytest.mark.parametrize("feature", ["flowcat", "gmm_patient", "gmm_file"])
def test_saved_seed_replays_features_across_processes(replayed_patient_features, feature):
    first, second = replayed_patient_features
    np.testing.assert_array_equal(first[feature], second[feature])


def test_per_tube_sampling_is_independent_of_file_location(tmp_path):
    original = tmp_path / "events.csv"
    values = np.arange(256, dtype=np.float32).reshape(128, 2)
    pd.DataFrame(values, columns=["CD19", "CD20"]).to_csv(original, index=False)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    expected = load_one_file(str(original), ["CD19", "CD20"], 8, 13)
    for index in range(3):
        copied = relocated / ("events_%s.csv" % index)
        shutil.copyfile(original, copied)
        observed = load_one_file(str(copied), ["CD19", "CD20"], 8, 13)
        np.testing.assert_array_equal(observed, expected)


def test_patient_table_preserves_explicit_tube_type(tmp_path):
    paths = [tmp_path / "acquisition_A.csv", tmp_path / "acquisition_B.csv"]
    for path in paths:
        pd.DataFrame({"CD19": [1.0], "CD20": [2.0]}).to_csv(path, index=False)
    workbook = tmp_path / "split.xlsx"
    pd.DataFrame(
        {
            "patient_id": ["synthetic_patient"] * 2,
            "csv_path": [str(path) for path in paths],
            "label": ["normal"] * 2,
            "tube_type": ["panel_0", "panel_1"],
        }
    ).to_excel(workbook, sheet_name="test", index=False)

    assert patient_table(str(workbook), "test")["_tube_id"].tolist() == ["panel_0", "panel_1"]


@pytest.mark.parametrize("changed_option", ["marker_columns", "class_to_index", "max_cells", "seed"])
def test_set_transformer_cache_respects_configuration(tmp_path, changed_option):
    csv_path = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "CD10": np.arange(32, dtype=np.float32),
            "CD19": np.arange(32, dtype=np.float32) + 100,
            "label": ["negative", "positive"] * 16,
        }
    ).to_csv(csv_path, index=False)
    initial = {
        "marker_columns": ["CD10", "CD19"],
        "class_to_index": {"negative": 0, "positive": 1},
        "max_cells": 8,
        "seed": 42,
    }
    changes = {
        "marker_columns": ["CD19", "CD10"],
        "class_to_index": {"negative": 1, "positive": 0},
        "max_cells": 2,
        "seed": 99,
    }
    shared_cache = str(tmp_path / "shared_cache")
    build_uniform_tube_cache(str(csv_path), cache_dir=shared_cache, **initial)
    requested = {**initial, changed_option: changes[changed_option]}
    expected_x, expected_y, expected_rows = build_uniform_tube_cache(
        str(csv_path), cache_dir=str(tmp_path / "fresh_cache"), **requested
    )
    observed_x, observed_y, observed_rows = build_uniform_tube_cache(
        str(csv_path), cache_dir=shared_cache, **requested
    )

    assert observed_rows == expected_rows
    np.testing.assert_array_equal(np.load(observed_x), np.load(expected_x))
    np.testing.assert_array_equal(np.load(observed_y), np.load(expected_y))


def test_pac_point_estimate_uses_full_test_data():
    targets = np.zeros(4000, dtype=np.int64)
    targets[:10] = 1
    scores = np.full(len(targets), 0.5)
    expected = average_precision_score(targets, scores)

    observed, _, _ = bootstrap_ci_binary(targets, scores, safe_prauc, n_boot=0, max_n=2000)

    assert observed == pytest.approx(expected)


def test_deepcytof_explicit_epochs_does_not_read_reference_dataset(tmp_path, monkeypatch):
    features = np.array([[-1.0], [1.0]], dtype=np.float32)
    targets = np.array([0, 1])
    _, workbook, mapping = write_cell_split(tmp_path, features, targets, ["negative", "positive"])
    monkeypatch.delenv("FCM_ML_MARKER_COLUMNS", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train", "--dataset_xlsx", str(workbook), "--class_mapping_path", str(mapping),
            "--result_root", str(tmp_path / "results"), "--deepcytof_epochs", "1",
        ],
    )

    def reject_reference_read(*args, **kwargs):
        pytest.fail("Explicit epochs must not read an unrelated reference workbook")

    calls = []

    def fake_training(*args, **kwargs):
        Path(args[4]).mkdir(parents=True, exist_ok=True)
        calls.append(kwargs["epochs"])
        return 0.0

    monkeypatch.setattr(_train, "count_train_samples_for_split", reject_reference_read)
    monkeypatch.setattr(_train, "train_eval_deepcytof", fake_training)

    _train.main("deepcytof")

    assert calls == [1]


@pytest.mark.parametrize("reader", ["ctransformer", "set_transformer"])
def test_cell_manifest_readers_honor_shared_path_map(tmp_path, monkeypatch, reader):
    csv_path = tmp_path / "events.csv"
    pd.DataFrame({"CD19": [1.0], "label": ["negative"]}).to_csv(csv_path, index=False)
    archived_prefix = str(tmp_path / "archived_location")
    workbook = tmp_path / "split.xlsx"
    pd.DataFrame({"csv_path": [archived_prefix + "/events.csv"]}).to_excel(
        workbook, sheet_name="test", index=False
    )
    monkeypatch.setenv("FCM_ML_PATH_MAP", json.dumps({archived_prefix: str(tmp_path)}))
    expected = collect_csv_paths_from_xlsx(str(workbook), sheet="test")
    assert expected == [str(csv_path)]

    if reader == "ctransformer":
        observed = _collect_test_csv_paths(str(workbook), csv_col="csv_path")
    else:
        observed = read_split_paths(str(workbook), "test")

    assert observed == expected
