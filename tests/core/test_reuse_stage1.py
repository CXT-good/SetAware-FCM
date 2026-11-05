from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cell_level.multicenter.reuse_stage1 import reuse_stage1_results


def test_reuse_requires_all_nine_pairs(tmp_path):
    with pytest.raises(FileNotFoundError, match="9 Stage1-only results"):
        reuse_stage1_results(tmp_path / "source", tmp_path / "target")


def test_reuse_copies_matrix_and_regenerates_metrics(tmp_path):
    source = tmp_path / "source"
    for train_center in ("bln", "bue", "vie14"):
        for test_center in ("bln", "bue", "vie14"):
            leaf = (
                source
                / f"train_{train_center}"
                / "stage1_best"
                / f"test_{test_center}"
                / "stage1"
            )
            leaf.mkdir(parents=True)
            np.save(leaf / "agg_confusion.npy", np.array([[3, 1], [1, 3]]))
            pd.DataFrame(
                {"class_name": ["0", "1", "overall"], "class_index": [0, 1, ""]}
            ).to_csv(leaf / "metrics_per_class.csv", index=False)
    target = tmp_path / "target"
    assert reuse_stage1_results(source, target) == 9
    leaf = target / "stage1_only" / "train_bln" / "test_bue"
    np.testing.assert_array_equal(
        np.load(leaf / "agg_confusion.npy"), np.array([[3, 1], [1, 3]])
    )
    assert (leaf / "overall_metrics.csv").is_file()
