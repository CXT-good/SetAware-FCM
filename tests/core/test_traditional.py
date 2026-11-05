from __future__ import annotations

import numpy as np

from cell_level.multicenter.common import COMMON_MARKERS
from cell_level.multicenter.traditional import (
    train_traditional_arrays,
    xgboost_params,
)


def test_lr_artifact_records_marker_order(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, len(COMMON_MARKERS))).astype(np.float32)
    y = np.array([0, 1] * 20, dtype=np.int64)
    train_traditional_arrays(
        "logistic_regression",
        x,
        y,
        ["A", "B"],
        tmp_path,
        marker_columns=COMMON_MARKERS,
        seed=42,
        max_iter=20,
    )
    data = np.load(tmp_path / "model.npz", allow_pickle=True)
    assert data["marker_columns"].tolist() == list(COMMON_MARKERS)
    assert data["class_list"].tolist() == ["A", "B"]


def test_xgboost_gpu_parameters():
    params = xgboost_params(3, 42)
    assert params["device"] == "cuda"
    assert params["tree_method"] == "hist"
    assert params["num_class"] == 3
    assert params["seed"] == 42
