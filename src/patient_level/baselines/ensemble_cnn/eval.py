from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from common.patient_paper import patient_labels, patient_table, predict_proba_cnn, save_predictions
from patient_level.baselines.ensemble_cnn.model import SmallHistogramCNN, build_images


def evaluate(
    split_xlsx: str,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    device: str = "cpu",
    batch_size: int = 128,
) -> dict:
    model_dir = Path(model_dir)
    saved = joblib.load(model_dir / "ensemble_cnn_paper.joblib")
    rows = patient_table(split_xlsx, split)
    labels = patient_labels(rows)
    patient_ids = labels["patient_id"].astype(str).tolist()
    encoder = saved["encoder"]
    feature_parts = []
    for index, spec in enumerate(saved["specs"], start=1):
        checkpoint = torch.load(model_dir / "pair_cnns" / f"pair_{index:04d}.pt", map_location=device, weights_only=False)
        model = SmallHistogramCNN(num_classes=len(encoder.classes_)).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        images = build_images(rows, patient_ids, spec, saved["args"]["bins"], saved["args"]["max_events"])
        feature_parts.append(predict_proba_cnn(model, images, device, batch_size))
    features = np.hstack(feature_parts)
    predictions = encoder.inverse_transform(saved["rf"].predict(features))
    probabilities = saved["rf"].predict_proba(features)
    return save_predictions(Path(output_dir), patient_ids, labels["label"].astype(str).to_numpy(), predictions, probabilities, encoder.classes_)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved marker-pair CNNs and their random-forest classifier.")
    parser.add_argument("--split_xlsx", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    print(evaluate(args.split_xlsx, args.model_dir, args.output_dir, split=args.split, device=args.device, batch_size=args.batch_size), flush=True)


if __name__ == "__main__":
    main()
