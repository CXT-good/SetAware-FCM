from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataio.marker_mapping import MARKER_LIST
from dataio.patient_dataset import PatientLevelDataset, build_label_maps, patient_collate, read_split_rows
from patient_level.ca_mil.model import PatientLevelMIL, build_encoder_from_checkpoint


MODEL_SIZE_PRESETS = {
    "tiny": dict(d_model=128, encoder_layers=6, decoder_layers=2, n_heads=8, d_ff=512),
    "small": dict(d_model=384, encoder_layers=8, decoder_layers=4, n_heads=16, d_ff=1536),
    "base": dict(d_model=512, encoder_layers=12, decoder_layers=4, n_heads=16, d_ff=2048),
}


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


def dist_setup(args: argparse.Namespace) -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        return 0, 1, 0, False
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    args.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    return rank, world_size, local_rank, True


def dist_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def gather_train_outputs(losses: list[float], y_true: list[int], y_pred: list[int], use_ddp: bool) -> tuple[list[float], list[int], list[int]]:
    if not use_ddp:
        return losses, y_true, y_pred
    gathered: list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, (losses, y_true, y_pred))
    all_losses: list[float] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    for part_losses, part_true, part_pred in gathered:
        all_losses.extend(part_losses)
        all_true.extend(part_true)
        all_pred.extend(part_pred)
    return all_losses, all_true, all_pred


def move_patient_to_device(sample: dict, device: torch.device) -> dict:
    for tube in sample["tubes"]:
        tube["values"] = tube["values"].to(device, non_blocking=True)
        tube["marker_ids"] = tube["marker_ids"].to(device, non_blocking=True)
        tube["row_ids"] = tube["row_ids"].to(device, non_blocking=True)
    return sample


def evaluate(model: PatientLevelMIL, loader: DataLoader, device: torch.device) -> dict:
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
    raw_model = unwrap_model(model)
    torch.save(
        {
            "format": "patient_level_mil_v1",
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
            "label_to_idx": label_to_idx,
            "model_state_dict": raw_model.state_dict(),
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
    p = argparse.ArgumentParser(description="Patient-level gated-attention MIL training.")
    p.add_argument("--split_xlsx", required=True)
    p.add_argument("--train_sheet", default="train")
    p.add_argument("--val_sheet", default="test")
    p.add_argument("--output_dir", default="outputs/patient_level")
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
    p.add_argument("--filter_mode", choices=["none", "count", "signature"], default="count")
    p.add_argument("--max_cells_per_tube", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=1, help="Patients per GPU. Variable patient bags currently support 1.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Patient-level bags have variable shape; use --batch_size 1 per GPU.")
    rank, world_size, local_rank, use_ddp = dist_setup(args)
    is_main = rank == 0
    preset = MODEL_SIZE_PRESETS[args.model_size]
    for k, v in preset.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    set_seed(args.seed + rank)

    train_rows = read_split_rows(args.split_xlsx, args.train_sheet)
    label_to_idx, idx_to_label = build_label_maps(train_rows["label"])
    device = torch.device(args.device)

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
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    ) if use_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=patient_collate,
        num_workers=0,
    )
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
    model = PatientLevelMIL(
        encoder,
        num_classes=len(label_to_idx),
        attention_hidden_dim=args.attention_hidden_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "label_map.json"), "w", encoding="utf-8") as f:
            json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f, ensure_ascii=False, indent=2)

    rows: list[dict] = []
    best_metric = -1.0
    best_epoch = 0
    best_path = os.path.join(args.output_dir, "checkpoint_best.pt")
    latest_path = os.path.join(args.output_dir, "checkpoint_latest.pt")

    try:
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            train_losses: list[float] = []
            train_true: list[int] = []
            train_pred: list[int] = []
            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", disable=not is_main)
            for sample in pbar:
                sample = move_patient_to_device(sample, device)
                target = torch.tensor([sample["label"]], dtype=torch.long, device=device)
                opt.zero_grad(set_to_none=True)
                logits = model(sample["tubes"])["logits"].unsqueeze(0)
                loss = criterion(logits, target)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                train_losses.append(float(loss.item()))
                train_true.append(int(target.item()))
                train_pred.append(int(logits.argmax(dim=1).item()))
                if is_main:
                    pbar.set_postfix(loss=f"{np.mean(train_losses):.4f}")

            all_losses, all_true, all_pred = gather_train_outputs(train_losses, train_true, train_pred, use_ddp)

            if is_main:
                val_metrics = evaluate(unwrap_model(model), val_loader, device)
                train_metrics = {
                    "loss": float(np.mean(all_losses)) if all_losses else 0.0,
                    "acc": float(accuracy_score(all_true, all_pred)) if all_true else 0.0,
                    "macro_f1": float(f1_score(all_true, all_pred, average="macro", zero_division=0)) if all_true else 0.0,
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
            if use_ddp:
                dist.barrier()

        if is_main:
            summary = TrainSummary(best_epoch, best_metric, best_path, latest_path, label_to_idx)
            with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
            print(asdict(summary))
    finally:
        dist_cleanup()


if __name__ == "__main__":
    main()
