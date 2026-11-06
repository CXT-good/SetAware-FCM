import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from common.patient_baselines import read_split_book
from common.patient_paper import patient_table
from patient_level.baselines.flowcat.model import patient_tensor, tube_order as flowcat_tube_order
from patient_level.baselines.gmm_fisher_svm.model_per_tube import fit_feature_pipeline
from patient_level.baselines.umap_rf.model import tube_order as umap_tube_order


@pytest.fixture
def patient_manifest(tmp_path):
    path = tmp_path / "patient_tube_2.csv"
    values = np.random.default_rng(4).normal(size=(64, 2)).astype(np.float32)
    pd.DataFrame(values, columns=["CD19", "CD20"]).to_csv(path, index=False)
    return pd.DataFrame({"patient_id": ["patient"], "csv_path": [str(path)], "label": ["normal"]})


@pytest.mark.parametrize("split_column", ["split", "set", "fold"])
def test_single_sheet_partition_values_ignore_case_and_whitespace(tmp_path, patient_manifest, split_column):
    frame = pd.concat([patient_manifest, patient_manifest], ignore_index=True)
    frame["patient_id"] = ["training_patient", "test_patient"]
    frame[split_column] = ["train", " TEST "]
    workbook = tmp_path / "split.xlsx"
    frame.to_excel(workbook, sheet_name="all_rows", index=False)

    rows = read_split_book(str(workbook), "test")

    assert rows["_patient_id"].tolist() == ["test_patient"]


@pytest.mark.parametrize("partitions", [
    {"fold": "fold_0", "set": "test"},
    {"set": "cohort_a", "fold": "fold_0", "split": "test"},
], ids=["set_before_fold", "split_before_set"])
def test_split_column_aliases_follow_declared_priority(tmp_path, patient_manifest, partitions):
    workbook = tmp_path / "split.xlsx"
    patient_manifest.assign(**partitions).to_excel(workbook, sheet_name="all_rows", index=False)

    assert read_split_book(str(workbook), "test")["_patient_id"].tolist() == ["patient"]


def test_missing_partition_in_single_sheet_raises(tmp_path, patient_manifest):
    frame = patient_manifest.assign(split="train")
    workbook = tmp_path / "split.xlsx"
    frame.to_excel(workbook, sheet_name="all_rows", index=False)

    with pytest.raises(ValueError):
        read_split_book(str(workbook), "test")


def test_multiple_sheets_do_not_supply_missing_split_from_first_sheet(tmp_path, patient_manifest):
    workbook = tmp_path / "split.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        patient_manifest.assign(split="test").to_excel(writer, sheet_name="train", index=False)
        patient_manifest.to_excel(writer, sheet_name="validation", index=False)

    with pytest.raises(ValueError):
        read_split_book(str(workbook), "test")


@pytest.mark.parametrize("requested", ["train", "test"])
def test_single_named_sheet_filters_explicit_mixed_partitions(tmp_path, patient_manifest, requested):
    frame = pd.concat([patient_manifest, patient_manifest], ignore_index=True)
    frame["patient_id"] = ["training_patient", "test_patient"]
    frame["split"] = ["train", "test"]
    workbook = tmp_path / "split.xlsx"
    frame.to_excel(workbook, sheet_name="train", index=False)

    expected = ["training_patient"] if requested == "train" else ["test_patient"]
    assert read_split_book(str(workbook), requested)["_patient_id"].tolist() == expected


def test_named_split_sheet_takes_precedence_in_multi_sheet_workbooks(tmp_path, patient_manifest):
    workbook = tmp_path / "split.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        patient_manifest.assign(set="cohort_a").to_excel(writer, sheet_name="TRAIN", index=False)
        patient_manifest.to_excel(writer, sheet_name="test", index=False)

    assert read_split_book(str(workbook), "train")["_patient_id"].tolist() == ["patient"]


def test_explicit_tube_type_falls_back_only_for_missing_values(tmp_path, patient_manifest):
    frame = pd.concat([patient_manifest] * 4, ignore_index=True)
    frame["tube_type"] = [" custom_panel ", None, " ", "panel_0"]
    workbook = tmp_path / "split.xlsx"
    frame.to_excel(workbook, sheet_name="test", index=False)

    assert patient_table(str(workbook), "test")["_tube_id"].tolist() == ["custom_panel", "2", "2", "panel_0"]


def test_numeric_tube_type_matches_saved_tube_across_splits(tmp_path, patient_manifest):
    train = patient_manifest.assign(tube_type=1)
    test = pd.concat([patient_manifest] * 2, ignore_index=True)
    test["patient_id"] = ["numeric_patient", "inferred_patient"]
    test["tube_type"] = [1, None]
    workbook = tmp_path / "split.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        train.to_excel(writer, sheet_name="train", index=False)
        test.to_excel(writer, sheet_name="test", index=False)
    train_rows = patient_table(str(workbook), "train")
    test_rows = patient_table(str(workbook), "test")
    saved_tubes = flowcat_tube_order(train_rows)

    assert saved_tubes == ["1"]
    assert test_rows["_tube_id"].tolist() == ["1", "2"]
    features = patient_tensor(test_rows, "numeric_patient", saved_tubes, 3, 64, 32, 13)
    assert features.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("column,identity", [
    ("tube_type", "01"),
    (" Tube_Type ", "1.0"),
    ("tube_type", 1.5),
])
def test_explicit_tube_identity_preserves_text_and_fractional_numbers(tmp_path, patient_manifest, column, identity):
    workbook = tmp_path / "split.xlsx"
    patient_manifest.assign(**{column: identity}).to_excel(workbook, sheet_name="test", index=False)

    assert patient_table(str(workbook), "test")["_tube_id"].tolist() == [str(identity)]


@pytest.mark.parametrize("order", [flowcat_tube_order, umap_tube_order], ids=["flowcat", "umap"])
def test_tube_order_supports_explicit_and_inferred_identities(order):
    rows = pd.DataFrame({"_tube_id": ["panel_0", "10", "2", "panel_0"]})

    assert order(rows) == ["2", "10", "panel_0"]


def test_gmm_feature_pipeline_supports_mixed_tube_identities(tmp_path, patient_manifest):
    frame = pd.concat([patient_manifest] * 2, ignore_index=True)
    frame["tube_type"] = ["panel_0", None]
    workbook = tmp_path / "split.xlsx"
    frame.to_excel(workbook, sheet_name="test", index=False)
    rows = patient_table(str(workbook), "test")
    args = argparse.Namespace(header_scan_limit=10, max_events_per_file=32,
                              max_events_per_patient_tube=16, max_gmm_events=32,
                              clip_quantile=0.01, seed=13)

    train, evaluated, pipes = fit_feature_pipeline(rows, ["patient"], rows, ["patient"], 1, args)

    assert [pipe["tube"] for pipe in pipes] == ["2", "panel_0"]
    assert train.shape == evaluated.shape == (1, 8)
    assert np.isfinite(train).all() and np.isfinite(evaluated).all()


@pytest.fixture(scope="module")
def replayed_per_tube_pipeline(tmp_path_factory):
    directory = tmp_path_factory.mktemp("per_tube_pipeline_replay")
    path = directory / "events.csv"
    values = np.random.default_rng(4).normal(size=(128, 2)).astype(np.float32)
    pd.DataFrame(values, columns=["CD19", "CD20"]).to_csv(path, index=False)
    program = """
import argparse
import json
import sys
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from patient_level.baselines.gmm_fisher_svm.model_per_tube import fit_tube_pipeline, transform_tube
rows = pd.DataFrame({'_patient_id': ['patient'], '_csv_path': [sys.argv[1]], '_tube_id': ['panel_0']})
args = argparse.Namespace(header_scan_limit=10, max_events_per_file=0,
    max_events_per_patient_tube=12, max_gmm_events=0, clip_quantile=0.01, seed=13)
fitted = fit_tube_pipeline(rows, 'panel_0', ['patient'], 1, args)
values = pd.read_csv(sys.argv[1]).to_numpy(dtype=np.float32)
scaler = StandardScaler().fit(values)
gmm = GaussianMixture(n_components=1, covariance_type='diag', random_state=13).fit(scaler.transform(values))
fixed = {'markers': ['CD19', 'CD20'], 'lo': np.array([-10., -10.]),
    'hi': np.array([10., 10.]), 'scaler': scaler, 'gmm': gmm, 'dim': 4}
features = transform_tube(rows, 'panel_0', ['patient'], fixed, args)
print(json.dumps({'fit': [fitted['lo'].tolist(), fitted['hi'].tolist(), fitted['gmm'].covariances_.tolist()],
    'transform': features.tolist()}))
"""
    results = []
    source_root = Path(__file__).resolve().parents[2] / "src"
    for hash_seed in (1, 2):
        environment = os.environ.copy()
        environment.update(PYTHONHASHSEED=str(hash_seed), PYTHONPATH=str(source_root),
                           PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
        result = subprocess.run([sys.executable, "-c", program, str(path)], cwd=directory,
                                env=environment, capture_output=True, text=True, check=True, timeout=60)
        results.append(json.loads(result.stdout.splitlines()[-1]))
    return results


@pytest.mark.parametrize("stage", ["fit", "transform"])
def test_per_tube_patient_sampling_replays_across_processes(replayed_per_tube_pipeline, stage):
    first, second = replayed_per_tube_pipeline

    assert first[stage] == second[stage]
