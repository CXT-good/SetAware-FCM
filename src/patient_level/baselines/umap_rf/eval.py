from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from common.patient_paper import patient_labels, patient_table, save_predictions
from patient_level.baselines.umap_rf.model import feature_matrix


def evaluate(
    split_xlsx: str,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
) -> dict:
    model_dir = Path(model_dir)
    saved = joblib.load(model_dir / "umap_rf_paper.joblib")
    reducers = {
        tube: joblib.load(model_dir / f"umap_tube_{tube}.joblib")["reducer"]
        for tube in saved["tubes"]
    }
    rows = patient_table(split_xlsx, split)
    labels = patient_labels(rows)
    patient_ids = labels["patient_id"].astype(str).tolist()
    features = feature_matrix(rows, patient_ids, saved["tubes"], reducers, saved["widths"], saved["args"]["bins"], saved["args"]["max_events"])
    encoder = saved["encoder"]
    predictions = encoder.inverse_transform(saved["rf"].predict(features))
    probabilities = saved["rf"].predict_proba(features)
    return save_predictions(Path(output_dir), patient_ids, labels["label"].astype(str).to_numpy(), predictions, probabilities, encoder.classes_)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved per-tube UMAP reducers and their random-forest classifier.")
    parser.add_argument("--split_xlsx", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    print(evaluate(args.split_xlsx, args.model_dir, args.output_dir, split=args.split), flush=True)


if __name__ == "__main__":
    main()
