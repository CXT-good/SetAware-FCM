from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.marker_mapping import MARKER_LIST
from dataio.patient_dataset import PatientLevelDataset, patient_collate
from patient_level.ca_mil.model import build_encoder_from_checkpoint
from patient_level.ca_mil.train import MODEL_SIZE_PRESETS, move_patient_to_device
from patient_level.taca_mil.model import PatientLevelTubeSetTransformerMIL


def fill_model_args(saved_args: dict) -> dict:
    args = dict(saved_args or {})
    preset = MODEL_SIZE_PRESETS.get(args.get("model_size", "tiny"), MODEL_SIZE_PRESETS["tiny"])
    for key, value in preset.items():
        if args.get(key) is None:
            args[key] = value
    args.setdefault("dropout", 0.1)
    args.setdefault("attention_hidden_dim", 128)
    args.setdefault("classifier_hidden_dim", 128)
    args.setdefault("set_hidden_dim", 128)
    args.setdefault("set_num_heads", 4)
    args.setdefault("set_num_inds", 16)
    args.setdefault("set_num_layers", 2)
    args.setdefault("filter_mode", "count")
    args.setdefault("max_cells_per_tube", 5000)
    args.setdefault("seed", 42)
    return args


def save_json(path: str | Path, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_confusion_matrix(path: str | Path, matrix: np.ndarray, labels: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix.tolist()):
            writer.writerow([label, *row])


def write_classification_report(path: str | Path, report: dict) -> None:
    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            row = {"class": label}
            row.update(metrics)
            rows.append(row)
        else:
            rows.append({"class": label, "value": metrics})
    fieldnames = sorted({key for row in rows for key in row.keys()})
    fieldnames = ["class"] + [key for key in fieldnames if key != "class"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate patient-level stage2 tube SetTransformer checkpoint.")
    p.add_argument("--split_xlsx", required=True)
    p.add_argument("--sheet", default="test")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--filter_mode", default=None, choices=["none", "count", "signature"])
    p.add_argument("--max_cells_per_tube", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def build_model(payload: dict, device: torch.device) -> tuple[PatientLevelTubeSetTransformerMIL, dict[str, int], dict]:
    label_to_idx = {str(k): int(v) for k, v in payload["label_to_idx"].items()}
    saved_args = fill_model_args(payload.get("args", {}))
    encoder = build_encoder_from_checkpoint(
        "",
        num_markers=len(MARKER_LIST),
        seq_len=len(MARKER_LIST),
        d_model=int(saved_args["d_model"]),
        encoder_layers=int(saved_args["encoder_layers"]),
        decoder_layers=int(saved_args["decoder_layers"]),
        n_heads=int(saved_args["n_heads"]),
        d_ff=int(saved_args["d_ff"]),
        dropout=float(saved_args["dropout"]),
        device=device,
    )
    model = PatientLevelTubeSetTransformerMIL(
        encoder,
        num_classes=len(label_to_idx),
        set_hidden_dim=int(saved_args["set_hidden_dim"]),
        set_num_heads=int(saved_args["set_num_heads"]),
        set_num_inds=int(saved_args["set_num_inds"]),
        set_num_layers=int(saved_args["set_num_layers"]),
        attention_hidden_dim=int(saved_args["attention_hidden_dim"]),
        classifier_hidden_dim=int(saved_args["classifier_hidden_dim"]),
        dropout=float(saved_args["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, label_to_idx, saved_args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, label_to_idx, saved_args = build_model(payload, device)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    labels = [idx_to_label[i] for i in range(len(idx_to_label))]
    # Evaluation-time data selection must follow the CLI arguments exactly.
    # Do not silently restore filter_mode/max_cells_per_tube/seed from the checkpoint,
    # because historical table-final evaluations may intentionally omit these options.
    filter_mode = "none" if args.filter_mode in (None, "none") else args.filter_mode
    max_cells_per_tube = args.max_cells_per_tube
    seed = args.seed

    ds = PatientLevelDataset(
        args.split_xlsx,
        args.sheet,
        label_to_idx=label_to_idx,
        filter_mode=filter_mode,
        max_cells_per_tube=max_cells_per_tube,
        random_sample=False,
        seed=seed,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=patient_collate, num_workers=args.num_workers)
    criterion = nn.CrossEntropyLoss(reduction="none")
    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []
    prediction_rows: list[dict] = []

    with torch.no_grad():
        for sample in tqdm(loader, desc=f"stage2-tube eval {Path(args.split_xlsx).stem}:{args.sheet}"):
            sample = move_patient_to_device(sample, device)
            target = torch.tensor([sample["label"]], dtype=torch.long, device=device)
            logits = model(sample["tubes"])["logits"].unsqueeze(0)
            loss = criterion(logits, target)
            prob = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
            pred_idx = int(np.argmax(prob))
            true_idx = int(target.item())
            y_true.append(true_idx)
            y_pred.append(pred_idx)
            losses.append(float(loss.item()))
            row = {
                "patient_id": sample["patient_id"],
                "true_label": idx_to_label[true_idx],
                "true_idx": true_idx,
                "pred_label": idx_to_label[pred_idx],
                "pred_idx": pred_idx,
                "correct": int(true_idx == pred_idx),
                "loss": float(loss.item()),
                "num_tubes": len(sample["tubes"]),
            }
            for class_idx, class_label in enumerate(labels):
                row[f"prob_{class_idx}_{class_label}"] = float(prob[class_idx])
            prediction_rows.append(row)

    acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if y_true else 0.0
    report = classification_report(y_true, y_pred, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0) if y_true else {}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))) if y_true else np.zeros((len(labels), len(labels)), dtype=int)

    if prediction_rows:
        with open(output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
            writer.writeheader()
            writer.writerows(prediction_rows)
    write_classification_report(output_dir / "metrics_per_class.csv", report)
    write_confusion_matrix(output_dir / "confusion_matrix.csv", cm, labels)

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_metrics": payload.get("metrics", {}),
        "split_xlsx": os.path.abspath(args.split_xlsx),
        "sheet": args.sheet,
        "output_dir": os.path.abspath(str(output_dir)),
        "device": str(device),
        "num_patients": len(y_true),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "labels": labels,
        "filter_mode": filter_mode,
        "max_cells_per_tube": max_cells_per_tube,
        "random_sample": False,
        "seed": seed,
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "eval_config.json", {"cli_args": vars(args), "checkpoint_args": saved_args, "label_to_idx": label_to_idx})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
