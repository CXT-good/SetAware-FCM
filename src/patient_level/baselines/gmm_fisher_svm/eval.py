from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from common.patient_paper import patient_table, write_json
from patient_level.baselines.gmm_fisher_svm.model import fisher_matrix, make_patient_cells, save_outputs
from patient_level.baselines.gmm_fisher_svm.model_per_tube import patient_ids_labels, transform_tube


def evaluate(
    split_xlsx: str,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    variant: str = "pooled",
    bootstrap: int | None = None,
) -> dict:
    if variant not in {"pooled", "per_tube"}:
        raise ValueError(f"Unknown GMM-Fisher-SVM variant: {variant}")
    filename = "gmm_fisher_svm_per_tube.joblib" if variant == "per_tube" else "gmm_fisher_svm.joblib"
    saved = joblib.load(Path(model_dir) / filename)
    args = argparse.Namespace(**saved["args"])
    rows = patient_table(split_xlsx, split)
    if variant == "per_tube":
        patient_ids, labels = patient_ids_labels(rows)
        feature_parts = [
            transform_tube(rows, pipe["tube"], patient_ids, pipe, args, seed_offset=77777 + index * 1000 + 200)
            for index, pipe in enumerate(saved["pipes"])
        ]
        features = np.hstack(feature_parts).astype(np.float32)
    else:
        patient_ids, labels, cells = make_patient_cells(rows, saved["markers"], args.max_cells_per_patient, args.seed + 10000)
        features = fisher_matrix(cells, saved["clip_low"], saved["clip_high"], saved["scaler"], saved["gmm"])
    encoder = saved["encoder"]
    predictions = encoder.inverse_transform(saved["svm"].predict(features))
    probabilities = saved["svm"].predict_proba(features)
    metrics = save_outputs(Path(output_dir), patient_ids, labels, predictions, probabilities, encoder.classes_, bootstrap if bootstrap is not None else args.bootstrap, args.seed)
    write_json(Path(output_dir) / "eval_meta.json", {"method": "gmm_fisher_svm", "variant": variant, "model_dir": str(model_dir), "split": split, "metrics": metrics})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved GMM-Fisher-SVM pipeline without fitting new components.")
    parser.add_argument("--split_xlsx", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--variant", choices=("pooled", "per_tube"), default="pooled")
    parser.add_argument("--bootstrap", type=int)
    args = parser.parse_args()
    print(evaluate(args.split_xlsx, args.model_dir, args.output_dir, split=args.split, variant=args.variant, bootstrap=args.bootstrap), flush=True)


if __name__ == "__main__":
    main()
