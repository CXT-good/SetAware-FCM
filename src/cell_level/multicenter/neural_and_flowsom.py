from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cell_level.multicenter.common import (
    COMMON_MARKERS,
    write_metric_artifacts,
)


def normalize_evaluation(
    method: str,
    raw_output_dir: str | Path,
    model_dir: str | Path,
    *,
    train_center: str,
    test_center: str,
) -> None:
    raw = Path(raw_output_dir)
    model_path = Path(model_dir)
    confusion = np.load(raw / "agg_confusion.npy")
    if method == "flowsom":
        meta = json.loads((model_path / "run_meta.json").read_text(encoding="utf-8"))
        class_list = list(meta["class_list"])
        markers = list(meta["features"])
        artifact = model_path / "flowsom_model.npz"
    elif method == "set_transformer_only":
        artifact = model_path if model_path.suffix == ".pt" else model_path / "checkpoint_best.pt"
        checkpoint = torch.load(artifact, map_location="cpu", weights_only=False)
        class_list = list(checkpoint["class_list"])
        markers = list(checkpoint["marker_columns"])
    else:
        raise ValueError(f"unsupported method: {method}")
    if tuple(markers) != COMMON_MARKERS:
        raise ValueError(f"artifact marker order is not the common panel: {markers}")
    write_metric_artifacts(
        confusion,
        class_list,
        raw,
        {
            "method": method,
            "train_center": train_center,
            "test_center": test_center,
            "model_artifact": str(artifact.resolve()),
            "marker_columns": markers,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize FlowSOM/ST-only outputs")
    parser.add_argument("--method", choices=["flowsom", "set_transformer_only"], required=True)
    parser.add_argument("--raw_output_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--train_center", required=True)
    parser.add_argument("--test_center", required=True)
    args = parser.parse_args()
    normalize_evaluation(
        args.method,
        args.raw_output_dir,
        args.model_dir,
        train_center=args.train_center,
        test_center=args.test_center,
    )


if __name__ == "__main__":
    main()
