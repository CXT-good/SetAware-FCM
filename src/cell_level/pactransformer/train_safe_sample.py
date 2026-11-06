from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.flow_dataset import TubeDataset
from cell_level.pactransformer.model import SetTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weighted Stage2 SetTransformer with OOM-safe per-tube subsampling")
    p.add_argument("--embeddings_dir", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--xuhao_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--dim_input", type=int, default=128)
    p.add_argument("--dim_hidden", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_inds", type=int, default=16)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--save_interval", type=int, default=5)
    p.add_argument("--max_cells_per_tube", type=int, default=120000)
    p.add_argument("--sample_seed", type=int, default=2024)
    return p.parse_args()


def compute_class_counts(labels_dir: Path, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    files = sorted(labels_dir.glob("*_labels.npy"))
    if not files:
        files = sorted(labels_dir.glob("*.npy"))
    for path in files:
        y = np.load(path).astype(np.int64).ravel()
        counts += np.bincount(y, minlength=num_classes)[:num_classes]
    if counts.sum() == 0:
        raise RuntimeError(f"No labels found under {labels_dir}")
    return counts


def class_weights_from_counts(counts: np.ndarray) -> np.ndarray:
    total = float(counts.sum())
    n = float(len(counts))
    w = np.array([total / (n * float(c)) if c > 0 else 0.0 for c in counts], dtype=np.float32)
    positive = w[w > 0]
    if positive.size:
        w = w / positive.mean()
    w[w == 0] = 1.0
    return w.astype(np.float32)


def save_checkpoint(path: Path, epoch: int, model, optimizer, args, counts, weights, avg_loss, avg_f1, best_loss):
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": int(args.num_classes),
            "dim_input": int(args.dim_input),
            "dim_hidden": int(args.dim_hidden),
            "num_heads": int(args.num_heads),
            "num_inds": int(args.num_inds),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "weighted_loss": True,
            "class_counts": [int(x) for x in counts.tolist()],
            "class_weights": [float(x) for x in weights.tolist()],
            "avg_loss": float(avg_loss),
            "avg_f1": float(avg_f1),
            "best_loss": float(best_loss),
            "max_cells_per_tube": int(args.max_cells_per_tube),
            "sample_seed": int(args.sample_seed),
            "config_note": "Weighted Stage2 training with per-tube stratified random subsampling to avoid CUDA OOM.",
        },
        path,
    )


def _subsample_batch(x: torch.Tensor, y: torch.Tensor, max_cells: int, rng: np.random.Generator):
    if max_cells <= 0 or x.shape[1] <= max_cells:
        return x, y, int(x.shape[1]), int(x.shape[1])

    yy = y[0].detach().cpu().numpy().astype(np.int64)
    classes = np.unique(yy)
    keep_parts = []

    # Keep a small guaranteed quota per class, then fill the remainder randomly.
    per_class_min = max(1, min(2000, max_cells // max(len(classes) * 4, 1)))
    for c in classes:
        idx = np.where(yy == c)[0]
        if idx.size:
            keep_parts.append(rng.choice(idx, size=min(idx.size, per_class_min), replace=False))

    keep = np.concatenate(keep_parts) if keep_parts else np.array([], dtype=np.int64)
    remaining = max_cells - keep.size
    if remaining > 0:
        mask = np.ones(len(yy), dtype=bool)
        if keep.size:
            mask[keep] = False
        rest = np.where(mask)[0]
        if rest.size:
            keep = np.concatenate([keep, rng.choice(rest, size=min(rest.size, remaining), replace=False)])
    rng.shuffle(keep)
    idx_t = torch.as_tensor(keep, dtype=torch.long)
    return x[:, idx_t, :], y[:, idx_t], int(keep.size), int(len(yy))


def train_epoch(model, loader, criterion, optimizer, device, max_cells_per_tube: int, rng: np.random.Generator):
    model.train()
    total_loss = 0.0
    total_f1 = 0.0
    total_n = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="Weighted Stage2 Training"), start=1):
        x = batch["embeddings"]
        y = batch["labels"].long()
        x, y, sampled_n, orig_n = _subsample_batch(x, y, int(max_cells_per_tube), rng)
        if sampled_n != orig_n:
            print(f"Batch {batch_idx}: sampled {sampled_n}/{orig_n} cells for OOM-safe training", flush=True)

        x = torch.nan_to_num(x.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        y = y.to(device).long()
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if not torch.isfinite(loss):
            print(f"Batch {batch_idx}: non-finite loss={loss.item()}, skipped to avoid corrupting weights", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        preds = logits.argmax(dim=-1).detach().cpu().numpy().ravel()
        yy = y.detach().cpu().numpy().ravel()
        f1 = f1_score(yy, preds, average="weighted", zero_division=0)
        precision = precision_score(yy, preds, average="weighted", zero_division=0)
        recall = recall_score(yy, preds, average="weighted", zero_division=0)
        n = int(yy.size)
        total_loss += float(loss.item()) * n
        total_f1 += float(f1) * n
        total_n += n
        print(f"Batch {batch_idx}: loss={loss.item():.6f}, f1={f1:.6f}, precision={precision:.6f}, recall={recall:.6f}", flush=True)
    return total_loss / max(total_n, 1), total_f1 / max(total_n, 1)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(f"max_cells_per_tube={args.max_cells_per_tube}", flush=True)

    counts = compute_class_counts(Path(args.labels_dir), args.num_classes)
    weights_np = class_weights_from_counts(counts)
    weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
    with (out / "class_weights.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "class_counts_from_stage2_train_labels": counts.tolist(),
                "class_weights": weights_np.tolist(),
                "max_cells_per_tube": int(args.max_cells_per_tube),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("class_counts_from_stage2_train_labels=", counts.tolist(), flush=True)
    print("class_weights=", weights_np.tolist(), flush=True)

    dataset = TubeDataset(args.embeddings_dir, args.labels_dir, args.xuhao_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = SetTransformer(
        dim_input=args.dim_input,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        num_inds=args.num_inds,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)
    print(f"model_params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    losses, f1s = [], []
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        avg_loss, avg_f1 = train_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
            args.max_cells_per_tube,
            np.random.default_rng(args.sample_seed + epoch),
        )
        scheduler.step()
        losses.append(float(avg_loss))
        f1s.append(float(avg_f1))
        np.save(out / "epoch_losses.npy", np.array(losses, dtype=np.float32))
        np.save(out / "epoch_f1s.npy", np.array(f1s, dtype=np.float32))
        save_checkpoint(out / "checkpoint_latest.pt", epoch, model, optimizer, args, counts, weights_np, avg_loss, avg_f1, min(best_loss, avg_loss))
        if avg_loss < best_loss:
            best_loss = float(avg_loss)
            save_checkpoint(out / "checkpoint_best.pt", epoch, model, optimizer, args, counts, weights_np, avg_loss, avg_f1, best_loss)
            print(f"saved new best: epoch={epoch}, loss={best_loss:.6f}", flush=True)
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            save_checkpoint(out / f"checkpoint_epoch_{epoch}.pt", epoch, model, optimizer, args, counts, weights_np, avg_loss, avg_f1, best_loss)
        print(f"Epoch {epoch} summary: avg_loss={avg_loss:.6f}, avg_f1={avg_f1:.6f}, best_loss={best_loss:.6f}", flush=True)
    print("weighted stage2 safe-sample training complete", flush=True)


if __name__ == "__main__":
    main()
