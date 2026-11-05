import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from cell_level.baselines import _eval, _train
from cell_level.pactransformer.eval import bootstrap_ci_binary, safe_prauc


def write_cell_inputs(directory, features, targets, class_list, dataset_key="synthetic"):
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"CD19": features})
    frame["label"] = [class_list[index] for index in targets]
    csv_path = directory / "events.csv"
    frame.to_csv(csv_path, index=False)
    workbook = directory / f"{dataset_key}_dataset_split.xlsx"
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
                    dataset_key: {
                        "class_list": class_list,
                        "class_counts": np.bincount(targets, minlength=len(class_list)).tolist(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return workbook, mapping


@pytest.mark.parametrize(
    ("coef", "classes", "targets", "expected_counts"),
    [
        ([[1.0]], [1, 3], [1, 1, 3], [0, 2, 0, 1, 0]),
        ([[1.0], [0.0], [-1.0]], [1, 2, 4], [4, 1, 1], [0, 2, 0, 0, 1]),
    ],
    ids=["binary-zero-score", "multiclass-tied-scores"],
)
def test_lr_checkpoint_restores_classes_and_score_ties(
    tmp_path, monkeypatch, coef, classes, targets, expected_counts
):
    monkeypatch.delenv("FCM_ML_MARKER_COLUMNS", raising=False)
    workbook, mapping = write_cell_inputs(
        tmp_path, [-1.0, 0.0, 1.0], targets, [f"class_{index}" for index in range(5)]
    )
    model_dir = tmp_path / "checkpoint"
    nested = model_dir / "model_outputs"
    nested.mkdir(parents=True)
    np.savez(
        nested / "model.npz",
        coef=np.asarray(coef),
        intercept=np.zeros(len(coef)),
        scale_mean=np.zeros(1),
        scale_std=np.ones(1),
        classes=np.asarray(classes),
    )
    output_dir = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval", "--model_dir", str(model_dir), "--dataset_xlsx", str(workbook),
            "--class_mapping_path", str(mapping), "--device", "cpu",
            "--output_dir", str(output_dir), "--warmup_batches", "0", "--batch_size", "2",
        ],
    )

    _eval.main("lr")

    observed = pd.read_csv(output_dir / "confusion_matrix.csv", index_col=0).to_numpy()
    np.testing.assert_array_equal(observed, np.diag(expected_counts))


@pytest.mark.parametrize("max_n", [1, 2, 2000])
def test_pac_ap_without_bootstrap_ignores_cap_and_nonfinite_scores(max_n):
    targets = np.array([1] * 10 + [0] * 3990 + [1, 0])
    scores = np.array([0.5] * 4000 + [np.nan, np.inf])

    point, low, high = bootstrap_ci_binary(
        targets, scores, safe_prauc, n_boot=0, max_n=max_n
    )

    assert point == pytest.approx(0.0025)
    assert np.isnan(low)
    assert np.isnan(high)


@pytest.mark.parametrize("max_n", [1, 2])
def test_pac_bootstrap_respects_small_balanced_cap(max_n):
    targets = np.array([0, 1] * 500)
    scores = np.full(len(targets), 0.5)

    point, low, high = bootstrap_ci_binary(
        targets, scores, safe_prauc, n_boot=30, max_n=max_n
    )

    assert point == pytest.approx(0.5)
    if max_n < 2:
        assert np.isnan(low)
        assert np.isnan(high)
    else:
        assert low == pytest.approx(0.5)
        assert high == pytest.approx(0.5)


@pytest.mark.parametrize("max_n", [1000, 2000])
def test_pac_capped_ap_interval_preserves_rare_class_prevalence(max_n):
    targets = np.array([1] * 10 + [0] * 3990)
    scores = np.full(len(targets), 0.5)

    point, low, high = bootstrap_ci_binary(
        targets, scores, safe_prauc, n_boot=500, max_n=max_n
    )

    # With tied scores AP is the positive prevalence, here 10 / 4000.
    # The capped interval may be wider, but must not shift up to an enriched
    # population whose prevalence is 10 / max_n.
    assert point == pytest.approx(0.0025)
    assert low < point < high
    assert high <= 0.006


@pytest.mark.parametrize("rare_class", [0, 1])
def test_pac_capped_interval_is_undefined_when_rounding_loses_a_class(rare_class):
    targets = np.array([rare_class] * 10 + [1 - rare_class] * 3990)
    scores = np.full(len(targets), 0.5)

    point, low, high = bootstrap_ci_binary(
        targets, scores, safe_prauc, n_boot=30, max_n=2
    )

    assert point == pytest.approx(0.0025 if rare_class == 1 else 0.9975)
    assert np.isnan(low)
    assert np.isnan(high)


@pytest.mark.parametrize(
    ("dataset_key", "requested_epochs", "epoch_cap", "expected_epochs", "reference_count"),
    [
        ("synthetic", 3, 2, 2, None),
        ("synthetic", 0, 0, 180, 8193),
        ("Zhongnan_Hospital", 0, 0, 60, 2),
    ],
    ids=["explicit-capped-no-reference", "automatic-reference", "automatic-self-reference"],
)
def test_deepcytof_records_only_reference_data_used_for_epoch_selection(
    tmp_path, monkeypatch, dataset_key, requested_epochs, epoch_cap,
    expected_epochs, reference_count
):
    monkeypatch.delenv("FCM_ML_MARKER_COLUMNS", raising=False)
    workbook, mapping = write_cell_inputs(
        tmp_path, [-1.0, 1.0], [0, 1], ["negative", "positive"], dataset_key
    )
    if reference_count == 8193:
        reference_csv = tmp_path / "reference_events.csv"
        pd.DataFrame({"CD19": np.zeros(reference_count)}).to_csv(reference_csv, index=False)
        pd.DataFrame({"csv_path": [str(reference_csv)]}).to_excel(
            tmp_path / "Zhongnan_Hospital_dataset_split.xlsx", sheet_name="train", index=False
        )

    def bounded_training(X_train, y_train, X_test, y_test, output_dir, **kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return kwargs["epochs"] / 1000

    monkeypatch.setattr(_train, "train_eval_deepcytof", bounded_training)
    result_root = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train", "--dataset_xlsx", str(workbook), "--class_mapping_path", str(mapping),
            "--result_root", str(result_root), "--deepcytof_epochs", str(requested_epochs),
            "--max_deepcytof_epochs", str(epoch_cap),
        ],
    )

    _train.main("deepcytof")

    dataset_results = result_root / dataset_key
    metadata = json.loads((dataset_results / "deepcytof" / "run_meta.json").read_text())
    summary = json.loads((dataset_results / "summary.json").read_text())
    assert metadata["deepcytof_epochs"] == expected_epochs
    assert summary["deepcytof"] == expected_epochs / 1000
    assert metadata["n_train_samples"] == 2
    assert metadata["steps_per_epoch"] == 1
    if requested_epochs > 0:
        assert metadata["epoch_mode"] == "explicit"
        for key in ("ref_dataset_key", "ref_epochs", "n_ref_train_samples", "target_total_steps"):
            assert metadata[key] is None
    else:
        assert metadata["epoch_mode"] == "automatic"
        assert metadata["ref_dataset_key"] == "Zhongnan_Hospital"
        assert metadata["ref_epochs"] == 60
        assert metadata["n_ref_train_samples"] == reference_count
        assert metadata["target_total_steps"] == expected_epochs
