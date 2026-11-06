from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.marker_mapping import MARKER_LIST
from dataio.patient_dataset import PatientLevelDataset, build_label_maps, patient_collate, read_split_rows
from patient_level.ca_mil.model import build_encoder_from_checkpoint
from patient_level.ca_mil.train import MODEL_SIZE_PRESETS, move_patient_to_device
from patient_level.taca_mil.model import PatientLevelTubeSetTransformerMIL


@dataclass
class TrainSummary:
    best_epoch: int
    best_metric: float
    best_path: str
    latest_path: str
    label_to_idx: dict[str, int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: PatientLevelTubeSetTransformerMIL, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for sample in loader:
            sample = move_patient_to_device(sample, device)
            target = torch.tensor([sample["label"]], dtype=torch.long, device=device)
            logits = model(sample["tubes"])["logits"].unsqueeze(0)
            loss = criterion(logits, target)
            losses.append(float(loss.item()))
            y_true.append(int(target.item()))
            y_pred.append(int(logits.argmax(dim=1).item()))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
    }


def save_checkpoint(path: str, model: nn.Module, args: argparse.Namespace, label_to_idx: dict[str, int], epoch: int, metrics: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "format": "patient_level_tube_set_transformer_v1",
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
            "label_to_idx": label_to_idx,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def write_metrics(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patient-level stage2 training: per-tube SetTransformer + patient MIL.")
    p.add_argument("--split_xlsx", required=True)
    p.add_argument("--train_sheet", default="train")
    p.add_argument("--val_sheet", default="test")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pretrained_encoder", default="")
    p.add_argument("--model_size", choices=sorted(MODEL_SIZE_PRESETS), default="tiny")
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--encoder_layers", type=int, default=None)
    p.add_argument("--decoder_layers", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--attention_hidden_dim", type=int, default=128)
    p.add_argument("--classifier_hidden_dim", type=int, default=128)
    p.add_argument("--set_hidden_dim", type=int, default=128)
    p.add_argument("--set_num_heads", type=int, default=4)
    p.add_argument("--set_num_inds", type=int, default=16)
    p.add_argument("--set_num_layers", type=int, default=2)
    p.add_argument("--filter_mode", choices=["none", "count", "signature"], default="count")
    p.add_argument("--max_cells_per_tube", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preset = MODEL_SIZE_PRESETS[args.model_size]
    for k, v in preset.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    set_seed(args.seed)
    device = torch.device(args.device)

    train_rows = read_split_rows(args.split_xlsx, args.train_sheet)
    label_to_idx, idx_to_label = build_label_maps(train_rows["label"])
    train_ds = PatientLevelDataset(
        args.split_xlsx,
        args.train_sheet,
        label_to_idx=label_to_idx,
        filter_mode=args.filter_mode,
        max_cells_per_tube=args.max_cells_per_tube,
        random_sample=True,
        seed=args.seed,
    )
    val_ds = PatientLevelDataset(
        args.split_xlsx,
        args.val_sheet,
        label_to_idx=label_to_idx,
        filter_mode=args.filter_mode,
        max_cells_per_tube=args.max_cells_per_tube,
        random_sample=False,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=patient_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=patient_collate, num_workers=0)

    encoder = build_encoder_from_checkpoint(
        args.pretrained_encoder,
        num_markers=len(MARKER_LIST),
        seq_len=len(MARKER_LIST),
        d_model=args.d_model,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        device=device,
    )
    model = PatientLevelTubeSetTransformerMIL(
        encoder,
        num_classes=len(label_to_idx),
        set_hidden_dim=args.set_hidden_dim,
        set_num_heads=args.set_num_heads,
        set_num_inds=args.set_num_inds,
        set_num_layers=args.set_num_layers,
        attention_hidden_dim=args.attention_hidden_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f, ensure_ascii=False, indent=2)

    rows: list[dict] = []
    best_metric = -1.0
    best_epoch = 0
    best_path = os.path.join(args.output_dir, "checkpoint_best.pt")
    latest_path = os.path.join(args.output_dir, "checkpoint_latest.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_true: list[int] = []
        train_pred: list[int] = []
        pbar = tqdm(train_loader, desc=f"stage2-tube epoch {epoch}/{args.epochs}")
        for sample in pbar:
            sample = move_patient_to_device(sample, device)
            target = torch.tensor([sample["label"]], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(sample["tubes"])["logits"].unsqueeze(0)
            loss = criterion(logits, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            train_losses.append(float(loss.item()))
            train_true.append(int(target.item()))
            train_pred.append(int(logits.argmax(dim=1).item()))
            pbar.set_postfix(loss=f"{np.mean(train_losses):.4f}")

        val_metrics = evaluate(model, val_loader, device)
        train_metrics = {
            "loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "acc": float(accuracy_score(train_true, train_pred)) if train_true else 0.0,
            "macro_f1": float(f1_score(train_true, train_pred, average="macro", zero_division=0)) if train_true else 0.0,
        }
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        rows.append(row)
        print(row, flush=True)
        save_checkpoint(latest_path, model, args, label_to_idx, epoch, row)
        if val_metrics["macro_f1"] >= best_metric:
            best_metric = val_metrics["macro_f1"]
            best_epoch = epoch
            save_checkpoint(best_path, model, args, label_to_idx, epoch, row)
        write_metrics(os.path.join(args.output_dir, "metrics.csv"), rows)

    summary = TrainSummary(best_epoch, best_metric, best_path, latest_path, label_to_idx)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
    print(asdict(summary), flush=True)


if __name__ == "__main__":
    main()
