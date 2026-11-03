from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cell_level.multicenter.common import write_metric_artifacts


CENTERS = ("bln", "bue", "vie14")
DEFAULT_SOURCE = Path(
    "outputs/multicenter/stage1_stage2_intersection_basepanel"
)
DEFAULT_TARGET = Path(
    "outputs/multicenter/baselines_intersection_basepanel_formal"
)


def _source_leaf(root: Path, train_center: str, test_center: str) -> Path:
    return (
        root
        / f"train_{train_center}"
        / "stage1_best"
        / f"test_{test_center}"
        / "stage1"
    )


def reuse_stage1_results(
    source_root: str | Path = DEFAULT_SOURCE,
    target_root: str | Path = DEFAULT_TARGET,
    *,
    validate_only: bool = False,
) -> int:
    source = Path(source_root)
    target = Path(target_root)
    pairs: list[tuple[str, str, Path]] = []
    missing: list[str] = []
    for train_center in CENTERS:
        for test_center in CENTERS:
            leaf = _source_leaf(source, train_center, test_center)
            required = (leaf / "agg_confusion.npy", leaf / "metrics_per_class.csv")
            absent = [str(path) for path in required if not path.is_file()]
            if absent:
                missing.extend(absent)
            else:
                pairs.append((train_center, test_center, leaf))
    if len(pairs) != 9:
        raise FileNotFoundError(
            f"expected 9 Stage1-only results, found {len(pairs)}; missing={missing}"
        )
    if validate_only:
        print("Stage1-only validation: 9/9 pairs available", flush=True)
        return 9

    for train_center, test_center, leaf in pairs:
        confusion = np.load(leaf / "agg_confusion.npy")
        metrics = pd.read_csv(
            leaf / "metrics_per_class.csv", dtype={"class_name": str}
        )
        class_names = metrics.loc[
            metrics["class_name"].str.lower() != "overall", "class_name"
        ].tolist()
        if len(class_names) != confusion.shape[0]:
            raise ValueError(
                f"class names do not match confusion matrix at {leaf}: "
                f"{len(class_names)} != {confusion.shape[0]}"
            )
        output = target / "stage1_only" / f"train_{train_center}" / f"test_{test_center}"
        write_metric_artifacts(
            confusion,
            class_names,
            output,
            {
                "method": "stage1_only",
                "train_center": train_center,
                "test_center": test_center,
                "reused": True,
                "source_output": str(leaf.resolve()),
            },
        )
    print("Stage1-only reuse complete: 9/9 pairs", flush=True)
    return 9


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize existing Stage1-only results")
    parser.add_argument("--source_root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--target_root", default=str(DEFAULT_TARGET))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    reuse_stage1_results(
        args.source_root, args.target_root, validate_only=args.validate_only
    )


if __name__ == "__main__":
    main()
