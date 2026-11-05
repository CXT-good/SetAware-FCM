from __future__ import annotations

import argparse
import importlib

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from common.patient_paper import SmallHistogramCNN, patient_table, predict_proba_cnn


@pytest.fixture
def patient_split(tmp_path):
    records = []
    for index, label in enumerate(["disease", "normal"]):
        values = np.linspace(0.0, 1.0, 32, dtype=np.float32)
        frame = pd.DataFrame({"CD19": values, "CD20": values if index == 0 else 1.0 - values})
        path = tmp_path / f"patient_{index}_tube_1.csv"
        frame.to_csv(path, index=False)
        records.append({"csv_path": str(path), "patient_id": f"p{index}", "label": label})
    split = tmp_path / "split.xlsx"
    with pd.ExcelWriter(split) as writer:
        pd.DataFrame(records).to_excel(writer, sheet_name="test", index=False)
    return split, patient_table(str(split), "test")


def _evaluation_module(method):
    name = f"patient_level.baselines.{method}.eval"
    assert importlib.util.find_spec(name) is not None, f"standalone evaluation is missing for {method}"
    return importlib.import_module(name)


def _assert_predictions(output, expected):
    frame = pd.read_csv(output / "test_predictions.csv")
    assert frame["patient_id"].tolist() == ["p0", "p1"]
    assert frame["label"].tolist() == ["disease", "normal"]
    assert frame["prediction"].tolist() == expected
    np.testing.assert_allclose(frame[["prob_disease", "prob_normal"]].sum(axis=1), 1.0, atol=1e-6)
    return frame


def test_ensemble_evaluation_restores_pair_cnns_and_random_forest(tmp_path, patient_split):
    split, rows = patient_split
    training = importlib.import_module("patient_level.baselines.ensemble_cnn.train")
    model_dir = tmp_path / "ensemble"
    pair_dir = model_dir / "pair_cnns"
    pair_dir.mkdir(parents=True)
    encoder = LabelEncoder().fit(["disease", "normal"])
    specs = [("1", "CD19", "CD20")]
    torch.manual_seed(7)
    model = SmallHistogramCNN(num_classes=2)
    images = training.build_images(rows, ["p0", "p1"], specs[0], 8, 32)
    features = predict_proba_cnn(model, images, "cpu")
    forest = RandomForestClassifier(n_estimators=8, bootstrap=False, random_state=3).fit(features, [0, 1])
    torch.save({"state_dict": model.state_dict(), "spec": specs[0], "classes": encoder.classes_.tolist()}, pair_dir / "pair_0001.pt")
    joblib.dump({"rf": forest, "encoder": encoder, "specs": specs, "args": {"bins": 8, "max_events": 32}}, model_dir / "ensemble_cnn_paper.joblib")

    output = tmp_path / "ensemble_eval"
    metrics = _evaluation_module("ensemble_cnn").evaluate(str(split), model_dir, output, device="cpu")

    assert metrics["accuracy"] == 1.0
    frame = _assert_predictions(output, ["disease", "normal"])
    np.testing.assert_array_equal(frame[["prob_disease", "prob_normal"]], [[1.0, 0.0], [0.0, 1.0]])


def test_flowcat_evaluation_restores_saved_cnn_and_feature_arguments(tmp_path, patient_split):
    split, rows = patient_split
    training = importlib.import_module("patient_level.baselines.flowcat.train")
    model_dir = tmp_path / "flowcat"
    model_dir.mkdir()
    torch.manual_seed(4)
    model = training.FlowCatCNN(in_channels=2, num_classes=2)
    args = {"grid_size": 3, "max_events": 32, "max_som_events": 32, "seed": 13, "batch_size": 1}
    tensors = np.stack([training.patient_tensor(rows, patient, ["1", "2"], 3, 32, 32, 13) for patient in ["p0", "p1"]])
    expected = training.predict(model, tensors, "cpu", 1)
    torch.save({"state_dict": model.state_dict(), "tubes": ["1", "2"], "classes": ["disease", "normal"], "args": args}, model_dir / "flowcat_paper.pt")

    output = tmp_path / "flowcat_eval"
    _evaluation_module("flowcat").evaluate(str(split), model_dir, output, device="cpu")

    frame = _assert_predictions(output, np.asarray(["disease", "normal"])[expected.argmax(axis=1)].tolist())
    np.testing.assert_allclose(frame[["prob_disease", "prob_normal"]], expected, rtol=1e-6)


def test_umap_evaluation_restores_reducers_and_saved_histogram_settings(tmp_path, patient_split):
    split, rows = patient_split
    training = importlib.import_module("patient_level.baselines.umap_rf.train")
    model_dir = tmp_path / "umap"
    model_dir.mkdir()
    reducer = PCA(n_components=2).fit(np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32))
    reducers = {"1": reducer, "2": reducer}
    widths = {"1": 2, "2": 2}
    features = training.feature_matrix(rows, ["p0", "p1"], ["1", "2"], reducers, widths, 4, 32)
    forest = RandomForestClassifier(n_estimators=8, bootstrap=False, random_state=3).fit(features, [0, 1])
    encoder = LabelEncoder().fit(["disease", "normal"])
    for tube in ["1", "2"]:
        joblib.dump({"reducer": reducer, "width": 2}, model_dir / f"umap_tube_{tube}.joblib")
    joblib.dump({"rf": forest, "encoder": encoder, "tubes": ["1", "2"], "widths": widths, "args": {"bins": 4, "max_events": 32}}, model_dir / "umap_rf_paper.joblib")

    output = tmp_path / "umap_eval"
    metrics = _evaluation_module("umap_rf").evaluate(str(split), model_dir, output)

    assert metrics["accuracy"] == 1.0
    _assert_predictions(output, ["disease", "normal"])


def test_gmm_evaluation_restores_preprocessing_fisher_vectors_and_svm(tmp_path, patient_split):
    split, rows = patient_split
    training = importlib.import_module("patient_level.baselines.gmm_fisher_svm.train")
    model_dir = tmp_path / "gmm"
    model_dir.mkdir()
    markers = ["CD19", "CD20"]
    ids, labels, cells = training.make_patient_cells(rows, markers, 32, 10013)
    low, high, scaler = training.fit_preprocess(cells, 0.01, 64, 32, 13)
    mixture = training.fit_gmm(cells, low, high, scaler, 2, 64, 32, 13)
    features = training.fisher_matrix(cells, low, high, scaler, mixture)
    encoder = LabelEncoder().fit(labels)
    classifier = SVC(kernel="linear", probability=True, random_state=13).fit(features, encoder.transform(labels))
    joblib.dump({"scaler": scaler, "clip_low": low, "clip_high": high, "gmm": mixture, "svm": classifier, "encoder": encoder, "markers": markers, "args": {"max_cells_per_patient": 32, "bootstrap": 0, "seed": 13}, "best": {"K": 2, "C": 1.0}}, model_dir / "gmm_fisher_svm.joblib")

    output = tmp_path / "gmm_eval"
    metrics = _evaluation_module("gmm_fisher_svm").evaluate(str(split), model_dir, output)

    assert metrics["accuracy"] == 1.0
    _assert_predictions(output, ["disease", "normal"])
    assert pd.read_csv(output / "overall_metrics.csv")["support"].tolist() == [2]
    assert pd.read_csv(output / "metrics_per_class.csv")["label"].tolist() == ["disease", "normal"]


def test_gmm_evaluation_restores_per_tube_pipelines(tmp_path, patient_split):
    split, rows = patient_split
    training = importlib.import_module("patient_level.baselines.gmm_fisher_svm.train_per_tube")
    model_dir = tmp_path / "gmm_per_tube"
    model_dir.mkdir()
    args = argparse.Namespace(header_scan_limit=10, max_events_per_file=32, max_events_per_patient_tube=32, max_gmm_events=64, clip_quantile=0.01, bootstrap=0, seed=13)
    features, _, pipes = training.fit_feature_pipeline(rows, ["p0", "p1"], rows, ["p0", "p1"], 2, args, seed_offset=77777)
    encoder = LabelEncoder().fit(["disease", "normal"])
    classifier = SVC(kernel="linear", probability=True, random_state=13).fit(features, [0, 1])
    joblib.dump({"pipes": pipes, "svm": classifier, "encoder": encoder, "args": vars(args), "best": {"K": 2, "C": 1.0}}, model_dir / "gmm_fisher_svm_per_tube.joblib")

    output = tmp_path / "gmm_per_tube_eval"
    metrics = _evaluation_module("gmm_fisher_svm").evaluate(str(split), model_dir, output, variant="per_tube")

    assert metrics["accuracy"] == 1.0
    _assert_predictions(output, ["disease", "normal"])
