from __future__ import annotations

import json

import numpy as np

from cell_level.baselines.flowsom.train import parse_args as flowsom_parse_args
from cell_level.baselines.flowsom.eval import parse_args as flowsom_eval_parse_args
from cell_level.multicenter.common import COMMON_MARKERS
from cell_level.multicenter.neural_and_flowsom import normalize_evaluation
from cell_level.baselines.set_transformer.train import parse_args as set_parse_args


def test_flowsom_parser_accepts_fixed_markers():
    args = flowsom_parse_args(
        [
            "--split_xlsx",
            "/tmp/source.xlsx",
            "--class_mapping_path",
            "/tmp/classes.json",
            "--dataset_key",
            "source",
            "--output_dir",
            "/tmp/out",
            "--marker_columns",
            *COMMON_MARKERS,
        ]
    )
    assert tuple(args.marker_columns) == COMMON_MARKERS


def test_flowsom_evaluator_can_skip_cell_dumps():
    args = flowsom_eval_parse_args(
        [
            "--model_dir",
            "/tmp/model",
            "--split_xlsx",
            "/tmp/test.xlsx",
            "--output_dir",
            "/tmp/output",
            "--skip_cell_dumps",
        ]
    )
    assert args.skip_cell_dumps is True


def test_set_transformer_parser_accepts_fixed_markers():
    args = set_parse_args(
        [
            "--train_xlsx",
            "/tmp/source.xlsx",
            "--class_mapping_path",
            "/tmp/classes.json",
            "--output_dir",
            "/tmp/out",
            "--marker_columns",
            *COMMON_MARKERS,
        ]
    )
    assert tuple(args.marker_columns) == COMMON_MARKERS


def test_normalize_evaluation_writes_standard_metrics(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    np.save(raw / "agg_confusion.npy", np.array([[3, 1], [1, 3]]))
    model = tmp_path / "model"
    model.mkdir()
    (model / "run_meta.json").write_text(
        json.dumps({"class_list": ["0", "1"], "features": list(COMMON_MARKERS)}),
        encoding="utf-8",
    )
    normalize_evaluation(
        "flowsom", raw, model, train_center="bln", test_center="bue"
    )
    assert (raw / "overall_metrics.csv").is_file()
    assert (raw / "eval_meta.json").is_file()
