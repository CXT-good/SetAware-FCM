import os
from pathlib import Path
import subprocess
import sys

import pytest

from cell_level.baselines import _eval, _train


BASELINES = [
    ("logistic_regression", "lr"),
    ("deepcytof", "deepcytof"),
    ("xgboost", "xgboost"),
]


@pytest.mark.parametrize(("model_name", "method"), BASELINES)
@pytest.mark.parametrize(
    ("entrypoint", "method_option"), [("train", "--methods"), ("eval", "--method")]
)
def test_model_entrypoint_help_binds_its_own_method(
    model_name, method, entrypoint, method_option, tmp_path
):
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(source_root), environment.get("PYTHONPATH", "")])
    )
    result = subprocess.run(
        [sys.executable, "-m", f"cell_level.baselines.{model_name}.{entrypoint}", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert f"{method_option} {{{method}}}" in result.stdout


@pytest.mark.parametrize(("model_name", "method"), BASELINES)
def test_training_parser_defaults_to_bound_model(model_name, method):
    args = _train._build_parser(method).parse_args(["--dataset_xlsx", "split.xlsx"])

    assert args.methods == [method]


@pytest.mark.parametrize(("model_name", "method"), BASELINES)
def test_evaluation_parser_defaults_to_bound_model(model_name, method):
    args = _eval._build_parser(method).parse_args(
        ["--model_dir", "checkpoint", "--dataset_xlsx", "split.xlsx"]
    )

    assert args.method == method


@pytest.mark.parametrize(("model_name", "method"), BASELINES)
@pytest.mark.parametrize(
    ("module", "method_option", "required_args"),
    [
        (_train, "--methods", ["--dataset_xlsx", "split.xlsx"]),
        (_eval, "--method", ["--model_dir", "checkpoint", "--dataset_xlsx", "split.xlsx"]),
    ],
)
def test_bound_parser_rejects_other_models(
    model_name, method, module, method_option, required_args
):
    for other_method in ["lr", "deepcytof", "xgboost"]:
        if other_method == method:
            continue
        with pytest.raises(SystemExit) as failure:
            module._build_parser(method).parse_args(
                [*required_args, method_option, other_method]
            )
        assert failure.value.code == 2


def test_generic_training_preserves_argument_defaults(monkeypatch, tmp_path):
    mapping_path = str(tmp_path / "classes.json")
    monkeypatch.setenv("FCM_ML_CLASS_MAPPING_PATH", mapping_path)
    args = _train._build_parser().parse_args(["--dataset_xlsx", "split.xlsx"])

    assert vars(args) == {
        "dataset_xlsx": "split.xlsx",
        "class_mapping_path": mapping_path,
        "result_root": "results",
        "max_cells_per_file": 50000,
        "methods": ["lr", "deepcytof", "xgboost"],
        "deepcytof_epochs": 0,
        "epochs_cache": "cache/dataset_train_steps_cache.json",
        "max_deepcytof_epochs": 0,
    }


def test_generic_evaluation_preserves_argument_defaults(monkeypatch, tmp_path):
    mapping_path = str(tmp_path / "classes.json")
    monkeypatch.setenv("FCM_ML_CLASS_MAPPING_PATH", mapping_path)
    args = _eval._build_parser().parse_args(
        ["--method", "lr", "--model_dir", "checkpoint", "--dataset_xlsx", "split.xlsx"]
    )

    assert vars(args) == {
        "method": "lr",
        "model_dir": "checkpoint",
        "dataset_xlsx": "split.xlsx",
        "output_root": "outputs",
        "output_dir": "",
        "class_mapping_path": mapping_path,
        "device": "cuda:0",
        "batch_size": 65536,
        "warmup_batches": 5,
        "timing_output": "inference_timing.json",
    }


def test_generic_evaluation_requires_a_method():
    with pytest.raises(SystemExit) as failure:
        _eval._build_parser().parse_args(
            ["--model_dir", "checkpoint", "--dataset_xlsx", "split.xlsx"]
        )

    assert failure.value.code == 2
