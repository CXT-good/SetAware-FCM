from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common.patient_paper import patient_labels, patient_table, save_predictions
from patient_level.baselines.flowcat.model import FlowCatCNN, patient_tensor, predict


def evaluate(
    split_xlsx: str,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    device: str = "cpu",
    batch_size: int | None = None,
) -> dict:
    saved = torch.load(Path(model_dir) / "flowcat_paper.pt", map_location=device, weights_only=False)
    args = saved["args"]
    rows = patient_table(split_xlsx, split)
    labels = patient_labels(rows)
    patient_ids = labels["patient_id"].astype(str).tolist()
    tensors = np.stack([
        patient_tensor(rows, patient_id, saved["tubes"], args["grid_size"], args["max_events"], args["max_som_events"], args["seed"])
        for patient_id in patient_ids
    ])
    classes = np.asarray(saved["classes"])
    model = FlowCatCNN(in_channels=len(saved["tubes"]), num_classes=len(classes)).to(device)
    model.load_state_dict(saved["state_dict"])
    probabilities = predict(model, tensors, device, batch_size if batch_size is not None else args["batch_size"])
    predictions = classes[np.argmax(probabilities, axis=1)]
    return save_predictions(Path(output_dir), patient_ids, labels["label"].astype(str).to_numpy(), predictions, probabilities, classes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved FlowCat CNN with its original SOM feature settings.")
    parser.add_argument("--split_xlsx", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    print(evaluate(args.split_xlsx, args.model_dir, args.output_dir, split=args.split, device=args.device, batch_size=args.batch_size), flush=True)


if __name__ == "__main__":
    main()
