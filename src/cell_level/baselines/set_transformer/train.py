from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from dataio.class_lists import dataset_key_from_split_xlsx
from cell_level.baselines.set_transformer.helpers import (
    StandaloneSetTransformer,
    build_uniform_tube_cache,
    class_info,
    infer_marker_columns,
    macro_f1_from_confusion,
    read_split_paths,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="直接用原始 marker 集合训练 Set Transformer")
    p.add_argument("--train_xlsx", required=True)
    p.add_argument("--class_mapping_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default="")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max_cells_per_tube", type=int, default=32768)
    p.add_argument("--csv_chunksize", type=int, default=65536)
    p.add_argument("--dim_hidden", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_inds", type=int, default=16)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--marker_columns", nargs="+", default=None)
    return p.parse_args(argv)


def save_checkpoint(path: str, *, model: nn.Module, optimizer, scheduler, epoch: int, args, meta) -> None:
    torch.save(
        {
            "format": "standalone_set_transformer_raw_markers_v1",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": meta["model_config"],
            "marker_columns": meta["marker_columns"],
            "class_list": meta["class_list"],
            "class_to_index": meta["class_to_index"],
            "dataset_key": meta["dataset_key"],
            "train_xlsx": os.path.abspath(args.train_xlsx),
            "max_cells_per_tube": args.max_cells_per_tube,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.output_dir, "raw_set_cache")

    paths = read_split_paths(args.train_xlsx, "train")
    info = class_info(args.train_xlsx, args.class_mapping_path)
    marker_columns = list(args.marker_columns) if args.marker_columns else infer_marker_columns(paths)
    print(f"dataset={info.dataset_key}, train_tubes={len(paths)}", flush=True)
    print(f"marker 数={len(marker_columns)}: {marker_columns}", flush=True)
    print(f"class 数={len(info.class_list)}: {info.class_list}", flush=True)

    cached: list[tuple[str, str]] = []
    sampled_counts = np.zeros(len(info.class_list), dtype=np.int64)
    for i, path in enumerate(paths, 1):
        print(f"准备训练集合 {i}/{len(paths)}", flush=True)
        x_path, y_path, _ = build_uniform_tube_cache(
            path,
            cache_dir=cache_dir,
            marker_columns=marker_columns,
            class_to_index=info.class_to_index,
            max_cells=args.max_cells_per_tube,
            seed=args.seed,
            chunksize=args.csv_chunksize,
        )
        cached.append((x_path, y_path))
        sampled_counts += np.bincount(
            np.load(y_path, mmap_mode="r"), minlength=len(info.class_list)
        )[: len(info.class_list)]

    nonzero = sampled_counts > 0
    weights = np.ones(len(info.class_list), dtype=np.float32)
    weights[nonzero] = sampled_counts[nonzero].sum() / (
        nonzero.sum() * sampled_counts[nonzero]
    )
    weights /= weights[nonzero].mean() if nonzero.any() else 1.0
    print(f"缓存样本类别数: {sampled_counts.tolist()}", flush=True)
    print(f"loss 类别权重: {weights.tolist()}", flush=True)

    model_config = {
        "dim_input": len(marker_columns),
        "dim_hidden": args.dim_hidden,
        "num_heads": args.num_heads,
        "num_inds": args.num_inds,
        "num_layers": args.num_layers,
        "num_classes": len(info.class_list),
        "dropout": args.dropout,
    }
    if args.dim_hidden % args.num_heads:
        raise ValueError("dim_hidden 必须能被 num_heads 整除")
    model = StandaloneSetTransformer(**model_config).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    meta = {
        "dataset_key": info.dataset_key,
        "class_list": info.class_list,
        "class_to_index": info.class_to_index,
        "marker_columns": marker_columns,
        "model_config": model_config,
    }
    write_json(
        os.path.join(args.output_dir, "run_meta.json"),
        {**meta, "args": vars(args), "sampled_class_counts": sampled_counts.tolist()},
    )

    start_epoch = 1
    best_f1 = -1.0
    history: list[dict] = []
    latest_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
    if args.resume and os.path.isfile(latest_path):
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        history_path = os.path.join(args.output_dir, "training_history.json")
        if os.path.isfile(history_path):
            with open(history_path, encoding="utf-8") as handle:
                history = json.load(handle)
            best_f1 = max((float(row["macro_f1"]) for row in history), default=-1.0)
        print(f"从 epoch {start_epoch} 继续", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        order = list(range(len(cached)))
        random.Random(args.seed + epoch).shuffle(order)
        cm = np.zeros((len(info.class_list), len(info.class_list)), dtype=np.int64)
        loss_sum = 0.0
        cells = 0
        epoch_start = time.time()
        for step, idx in enumerate(order, 1):
            x_path, y_path = cached[idx]
            x_np = np.array(np.load(x_path, mmap_mode="r"), copy=True)
            y_np = np.array(np.load(y_path, mmap_mode="r"), copy=True)
            permutation = np.random.default_rng(args.seed + epoch * 1000003 + idx).permutation(len(y_np))
            x = torch.from_numpy(x_np[permutation]).unsqueeze(0).to(device)
            y = torch.from_numpy(y_np[permutation]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x).squeeze(0)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            pred = logits.detach().argmax(dim=-1).cpu().numpy()
            cm += confusion_matrix(y_np[permutation], pred, labels=range(len(info.class_list)))
            loss_sum += float(loss.detach().cpu()) * len(y_np)
            cells += len(y_np)
            if step == 1 or step % 20 == 0 or step == len(order):
                print(
                    f"epoch {epoch}/{args.epochs} step {step}/{len(order)} "
                    f"loss={float(loss.detach().cpu()):.6f}",
                    flush=True,
                )
        scheduler.step()
        epoch_f1 = macro_f1_from_confusion(cm, info.class_list, info.dataset_key)
        row = {
            "epoch": epoch,
            "loss": loss_sum / max(1, cells),
            "macro_f1": epoch_f1,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_start,
            "cells": cells,
        }
        history.append(row)
        write_json(os.path.join(args.output_dir, "training_history.json"), history)
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            args=args,
            meta=meta,
        )
        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            save_checkpoint(
                os.path.join(args.output_dir, "checkpoint_best.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                args=args,
                meta=meta,
            )
        print(f"epoch summary: {row}, best_macro_f1={best_f1:.6f}", flush=True)


if __name__ == "__main__":
    main()
