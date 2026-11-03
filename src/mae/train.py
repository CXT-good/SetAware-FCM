from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import asdict
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from mae.layout import (
    ensure_experiment_layout,
    ensure_parent_dir,
    pretrain_encoder_ckpt_name,
    pretrain_mae_full_ckpt_name,
)
from dataio.marker_mapping import MARKER_LIST
from mae.model import MAEConfig, MAEModel, sample_mask
from dataio.mae_dataset import PretrainEpochBatchIterableDataset


def _dist_setup() -> tuple[int, int, int]:
    """多卡 DDP：返回 (rank, world_size, local_rank)。须由 torchrun 启动且 WORLD_SIZE>=2。"""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws < 2:
        raise RuntimeError(
            "本脚本仅支持多卡 DDP。请使用 torchrun 启动，例如：\n"
            "  torchrun --standalone --nproc_per_node=2 -m mae.train --pretrain_manifest <xlsx> ..."
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    # 默认 NCCL 集合通信超时是 600s（10 分钟），出问题时所有 rank 要空转 10 分钟才能感知。
    # 这里降到 120s，可通过环境变量 NCCL_COLLECTIVE_TIMEOUT_SEC 覆盖。
    timeout_sec = int(os.environ.get("NCCL_COLLECTIVE_TIMEOUT_SEC", "120"))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_sec))
    return rank, ws, local_rank


def _dist_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _unwrap(model: DDP) -> nn.Module:
    return model.module


# MAE 预训练模型规模预设（可通过 --model_size 一键选用；单项 --d_model 等可覆盖预设）
MODEL_SIZE_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {
        "d_model": 128,
        "encoder_layers": 6,
        "decoder_layers": 2,
        "n_heads": 8,
        "d_ff": 512,
    },
    "small": {
        "d_model": 384,
        "encoder_layers": 8,
        "decoder_layers": 4,
        "n_heads": 16,
        "d_ff": 1536,
    },
    "base": {
        "d_model": 512,
        "encoder_layers": 12,
        "decoder_layers": 4,
        "n_heads": 16,
        "d_ff": 2048,
    },
}


def _apply_model_size_preset(args: argparse.Namespace) -> None:
    preset = MODEL_SIZE_PRESETS[args.model_size]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def _estimate_total_rows(
    csv_files: list[str],
    sample_n: int = 150,
    *,
    max_rows_per_file: int = 0,
) -> float:
    """抽样估计 manifest 总细胞行数，用于推算总 step 数。
    若 max_rows_per_file>0，则每个文件按截断后的行数计算。"""
    if not csv_files:
        return 0.0
    rng = random.Random(0)
    sample = rng.sample(csv_files, min(sample_n, len(csv_files)))
    rows: list[int] = []
    cap = int(max_rows_per_file) if max_rows_per_file else 0
    for path in sample:
        try:
            with open(path, "rb") as f:
                n = max(sum(1 for _ in f) - 1, 0)
            if cap > 0:
                n = min(n, cap)
            rows.append(n)
        except OSError:
            continue
    if not rows:
        return 0.0
    return (sum(rows) / len(rows)) * len(csv_files)


def _estimate_training_schedule(
    csv_files: list[str],
    *,
    batch_size: int,
    world_size: int,
    epochs: int,
    max_steps: int,
    max_rows_per_file: int = 0,
) -> tuple[int, int]:
    """返回 (total_steps, steps_per_epoch)。"""
    total_rows = _estimate_total_rows(csv_files, max_rows_per_file=max_rows_per_file)
    global_batch = batch_size * world_size
    if total_rows <= 0 or global_batch <= 0:
        steps_per_epoch = 1
    else:
        steps_per_epoch = max(1, int(total_rows / global_batch))
    total = steps_per_epoch * max(epochs, 1)
    if max_steps > 0:
        total = min(total, max_steps)
    return max(total, 1), steps_per_epoch


def _resolve_warmup_steps(*, total_steps: int, warmup_ratio: float) -> int:
    if warmup_ratio <= 0:
        return 0
    n = int(total_steps * warmup_ratio)
    return min(max(n, 0), max(total_steps - 1, 0))


def _build_lr_scheduler(
    opt: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    peak_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Linear warmup (base→peak) + cosine decay (peak→min)，按 step 执行。"""
    from torch.optim.lr_scheduler import LambdaLR

    if scheduler_name == "none":
        return None

    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            progress = step / max(warmup_steps, 1)
            lr = base_lr + (peak_lr - base_lr) * progress
        else:
            t = min((step - warmup_steps) / decay_steps, 1.0)
            lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * t))
        return lr / peak_lr if peak_lr > 0 else 1.0

    return LambdaLR(opt, lr_lambda=lr_lambda)


def _checkpoint_stem(save_path: str) -> str:
    root, ext = os.path.splitext(save_path)
    return root, ext or ".pt"


def _train_checkpoint_path(save_path: str, epoch: int) -> str:
    stem, ext = _checkpoint_stem(save_path)
    return f"{stem}_train_epoch{epoch}{ext}"


def _train_latest_checkpoint_path(save_path: str) -> str:
    stem, ext = _checkpoint_stem(save_path)
    return f"{stem}_train_latest{ext}"


def _encoder_epoch_checkpoint_path(save_path: str, epoch: int) -> str:
    stem, ext = _checkpoint_stem(save_path)
    return f"{stem}_epoch{epoch}{ext}"


def _save_training_checkpoint(
    *,
    path: str,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    cfg: MAEConfig,
    model_size: str,
    epoch: int,
    step: int,
    loss_history: list[float],
    log_rows: list[tuple[int, int, int, float, float, float]],
) -> None:
    raw = model.module if isinstance(model, DDP) else model
    payload: dict = {
        "format": "pretrain_train_ckpt_v1",
        "cfg": asdict(cfg),
        "model_size": model_size,
        "epoch": epoch,
        "step": step,
        "state_dict": raw.state_dict(),
        "encoder_state_dict": raw.encoder.state_dict(),
        "optimizer": opt.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "loss_history": loss_history,
        "log_rows": log_rows,
    }
    ensure_parent_dir(path)
    torch.save(payload, path)


def _load_weights_into_model(model: nn.Module, payload: dict) -> None:
    raw = model.module if isinstance(model, DDP) else model
    if "state_dict" in payload:
        keys = payload["state_dict"].keys()
        if keys and all(k.startswith("encoder.") for k in keys):
            raw.encoder.load_state_dict(payload["state_dict"], strict=False)
        else:
            raw.load_state_dict(payload["state_dict"], strict=False)
    elif "encoder_state_dict" in payload:
        raw.encoder.load_state_dict(payload["encoder_state_dict"], strict=False)
    else:
        raise KeyError("checkpoint 中缺少 state_dict / encoder_state_dict")


def _resolve_resume_path(resume_arg: str, save_path: str) -> str:
    if resume_arg in ("", "latest"):
        return _train_latest_checkpoint_path(save_path)
    return os.path.abspath(resume_arg)


def _dist_should_skip_step(*, loss: torch.Tensor, device: torch.device) -> bool:
    """任一 rank 的 loss 非有限则全体 skip，避免 DDP backward 次数不一致。"""
    local_ok = torch.tensor(
        [1 if bool(torch.isfinite(loss).all()) else 0],
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
    return int(local_ok.item()) == 0


def _dist_should_skip_after_backward(*, model: DDP, device: torch.device) -> bool:
    """任一 rank 出现非有限梯度则全体 skip optimizer.step。"""
    local_ok = 1
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            local_ok = 0
            break
    flag = torch.tensor([local_ok], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return int(flag.item()) == 0


def main() -> None:
    p = argparse.ArgumentParser(description="MAE 预训练（仅多卡 DDP，请用 torchrun 启动）")
    p.add_argument("--epochs", type=int, default=5, help="训练轮数（Epoch 外层循环）")
    p.add_argument("--steps", type=int, default=0, help="可选：最多训练多少步（0 表示不限制，以 epochs 为准）")
    p.add_argument(
        "--batch_size",
        type=int,
        default=6144,
        help="每个 GPU 上的 batch 大小（全局等效 batch ≈ batch_size × GPU 数）",
    )
    p.add_argument("--lr", type=float, default=1e-5, help="峰值学习率（warmup 结束、余弦衰减起点）")
    p.add_argument(
        "--warmup_start_lr",
        type=float,
        default=5e-7,
        help="warmup 起始学习率",
    )
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine"],
        help="cosine=warmup 线性爬坡 + 余弦衰减至 min_lr（默认）",
    )
    p.add_argument("--min_lr", type=float, default=1e-6, help="训练结束时的目标学习率")
    p.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.2,
        help="warmup 占总训练步数比例（默认 1/5 线性爬坡，剩余步数余弦衰减）",
    )
    p.add_argument("--data_dir", type=str, default="", help="无标签 CSV 所在文件夹（与 --pretrain_manifest 二选一）。")
    p.add_argument(
        "--pretrain_manifest",
        type=str,
        default="",
        help="预训练 CSV 清单 xlsx（需含 csv_path 列），例如 data/pretrain_manifest.xlsx",
    )
    p.add_argument("--csv_recursive", action="store_true", help="递归搜索 data_dir 下所有 CSV")
    p.add_argument("--csv_max_files", type=int, default=0, help="限制最多读取多少个 CSV（0 表示不限制）")
    p.add_argument("--csv_seed", type=int, default=0)
    p.add_argument(
        "--max_rows_per_file",
        type=int,
        default=0,
        help="单 CSV 最多读多少行（0=不限）。"
        "数据集里有少量百万行级超大文件会让单个 worker 加载就吃 >2GB CPU 内存，"
        "并叠加触发 GPU 显存碎片化 OOM；建议 300000 左右兼顾覆盖率与稳定性。",
    )
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader worker 数（建议 8）")
    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="每个 worker 预取的 batch 数（num_workers>0 时生效，默认 4）",
    )
    p.add_argument(
        "--model_size",
        type=str,
        default="tiny",
        choices=sorted(MODEL_SIZE_PRESETS),
        help="模型规模预设：tiny（默认）/ small / base；见 MODEL_SIZE_PRESETS",
    )
    p.add_argument("--d_model", type=int, default=None, help="覆盖预设：隐藏维度")
    p.add_argument("--encoder_layers", type=int, default=None, help="覆盖预设：encoder 层数")
    p.add_argument("--decoder_layers", type=int, default=None, help="覆盖预设：decoder 层数")
    p.add_argument("--n_heads", type=int, default=None, help="覆盖预设：注意力头数")
    p.add_argument("--d_ff", type=int, default=None, help="覆盖预设：FFN 中间维")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mask_ratio", type=float, default=0.25)
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="梯度裁剪 max_norm（越小越强，默认 0.5；原 1.0）",
    )
    p.add_argument("--experiment_dir", type=str, default="", help="实验输出目录；默认权重写入 pretrained_<model_size>.pt。")
    p.add_argument("--save_path", type=str, default="", help="encoder 权重路径；不设时默认为 pretrained_<model_size>.pt。")
    p.add_argument("--log_path", type=str, default="", help="训练日志 CSV 输出路径（不传则自动生成）")
    p.add_argument("--loss_curve_path", type=str, default="", help="loss 曲线图片输出路径（不传则自动生成）")
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=2,
        help="每 N 个 epoch 保存可 resume 的训练 checkpoint（含 optimizer/scheduler）",
    )
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="从训练 checkpoint 恢复（路径或 latest）；恢复 epoch/step/optimizer/日志。与 --load_checkpoint 互斥时优先 resume。",
    )
    p.add_argument(
        "--load_checkpoint",
        type=str,
        default="",
        help="仅加载模型权重，从 epoch 1、step 0 重新训练（不恢复 optimizer）",
    )
    args = p.parse_args()
    if args.resume and args.load_checkpoint:
        raise ValueError("请只指定 --resume 或 --load_checkpoint 之一")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint_every 须 >= 1")
    _apply_model_size_preset(args)

    rank, world_size, local_rank = _dist_setup()
    is_main = rank == 0

    if args.experiment_dir:
        layout = ensure_experiment_layout(args.experiment_dir)
        if not args.save_path:
            args.save_path = os.path.join(
                layout["root"], pretrain_encoder_ckpt_name(args.model_size)
            )
    elif not args.save_path:
        args.save_path = pretrain_encoder_ckpt_name(args.model_size)

    if is_main:
        ensure_parent_dir(args.save_path)

    if not args.data_dir and not args.pretrain_manifest:
        raise ValueError("请提供 --data_dir 或 --pretrain_manifest（预训练 CSV 清单 xlsx）")

    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        print(f"使用设备: {device}  (DDP x{world_size})")
        print(
            f"模型规模: {args.model_size}  "
            f"(d_model={args.d_model}, enc={args.encoder_layers}, dec={args.decoder_layers}, "
            f"n_heads={args.n_heads}, d_ff={args.d_ff})"
        )
        max_rows_str = (
            f"{int(args.max_rows_per_file):,}/文件 截断" if int(args.max_rows_per_file) > 0 else "不限"
        )
        print(
            f"DataLoader: num_workers={args.num_workers}, prefetch_factor={args.prefetch_factor}, "
            f"persistent_workers=True, 每 epoch 复用单个 loader; max_rows_per_file={max_rows_str}"
        )
        print(
            f"训练稳定性: max_grad_norm={args.max_grad_norm}, "
            f"非有限 loss/梯度时 skip step（DDP 全 rank 同步）; "
            f"每 epoch 固定 steps_per_epoch 步（dataloader 耗尽自动循环），保证各 rank collective 严格对齐",
            flush=True,
        )

    marker_list = list(MARKER_LIST)
    if not marker_list:
        raise ValueError("marker_list 为空：请在 src/data2vec_flow/marker_mapping.py 中补全 MARKER_LIST。")

    cfg = MAEConfig(
        num_markers=len(marker_list),
        seq_len=len(marker_list),
        d_model=args.d_model,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
    )

    extreme_files_to_skip = {
        "wangxiqi_Tube_001_024_with_gates.csv",
        "lirong_Tube_001_010_with_gates.csv",
        "chenhuarong_Tube_001_019_with_gates.csv",
    }

    def _list_csv_files_from_manifest(manifest_path: str, *, max_files: int) -> list[str]:
        import pandas as pd

        df = pd.read_excel(manifest_path)
        if "csv_path" not in df.columns:
            raise ValueError(f"{manifest_path} 需包含 csv_path 列，当前列: {list(df.columns)}")
        out: list[str] = []
        missing = 0
        for raw in df["csv_path"].astype(str):
            p = os.path.abspath(raw.strip())
            if not p.lower().endswith(".csv"):
                continue
            if os.path.isfile(p):
                out.append(p)
            else:
                missing += 1
            if max_files and len(out) >= max_files:
                break
        if missing and is_main:
            print(f"警告: manifest 中有 {missing} 个 csv_path 在磁盘上不存在（已跳过）")
        return out

    def _list_csv_files(data_dir: str, *, recursive: bool, max_files: int) -> list[str]:
        out: list[str] = []
        skipped_extreme = 0
        if recursive:
            for root, _dirs, files in os.walk(data_dir):
                for f in files:
                    if f.lower().endswith(".csv"):
                        if f in extreme_files_to_skip:
                            skipped_extreme += 1
                            continue
                        out.append(os.path.join(root, f))
                        if max_files and len(out) >= max_files:
                            if skipped_extreme and is_main:
                                print(f"警告: 跳过了 {skipped_extreme} 个极端值CSV文件")
                            return sorted(out)
        else:
            for f in os.listdir(data_dir):
                if f.lower().endswith(".csv"):
                    if f in extreme_files_to_skip:
                        skipped_extreme += 1
                        continue
                    out.append(os.path.join(data_dir, f))
                    if max_files and len(out) >= max_files:
                        if skipped_extreme and is_main:
                            print(f"警告: 跳过了 {skipped_extreme} 个极端值CSV文件")
                        return sorted(out)
        if skipped_extreme and is_main:
            print(f"警告: 跳过了 {skipped_extreme} 个极端值CSV文件")
        return sorted(out)

    if args.pretrain_manifest:
        csv_files = _list_csv_files_from_manifest(
            args.pretrain_manifest, max_files=args.csv_max_files
        )
        if is_main:
            print(f"预训练清单: {args.pretrain_manifest}  -> {len(csv_files)} 个 CSV")
    else:
        csv_files = _list_csv_files(
            args.data_dir, recursive=bool(args.csv_recursive), max_files=args.csv_max_files
        )
    if not csv_files:
        src = args.pretrain_manifest or args.data_dir
        raise ValueError(f"未找到任何可训练的 CSV: {src}")

    def _build_epoch_dataloader(file_order: list[int], epoch: int) -> DataLoader:
        ds = PretrainEpochBatchIterableDataset(
            csv_files=csv_files,
            file_order=file_order,
            marker_columns=marker_list,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
            seed=args.csv_seed,
            verbose=is_main,
            max_rows_per_file=int(args.max_rows_per_file),
        )
        loader_kwargs: dict = {
            "batch_size": None,
            "pin_memory": True,
            "num_workers": args.num_workers,
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
        return DataLoader(ds, **loader_kwargs)

    model = DDP(
        MAEModel(cfg).to(device),
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    peak_lr = float(args.lr)
    base_lr = float(args.warmup_start_lr)
    min_lr = float(args.min_lr)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr)
    total_train_steps, steps_per_epoch = _estimate_training_schedule(
        csv_files,
        batch_size=args.batch_size,
        world_size=world_size,
        epochs=int(args.epochs),
        max_steps=int(args.steps),
        max_rows_per_file=int(args.max_rows_per_file),
    )
    warmup_steps = _resolve_warmup_steps(
        total_steps=total_train_steps,
        warmup_ratio=float(args.warmup_ratio),
    )
    lr_scheduler = _build_lr_scheduler(
        opt,
        scheduler_name=args.lr_scheduler,
        total_steps=total_train_steps,
        warmup_steps=warmup_steps,
        base_lr=base_lr,
        peak_lr=peak_lr,
        min_lr=min_lr,
    )
    if is_main:
        if lr_scheduler is None:
            print(f"学习率: 固定 lr={peak_lr}", flush=True)
        else:
            warmup_epochs_est = warmup_steps / max(steps_per_epoch, 1)
            decay_epochs_est = (total_train_steps - warmup_steps) / max(steps_per_epoch, 1)
            print(
                f"学习率: 前 {args.warmup_ratio:.0%} 步 warmup {base_lr:.1e}→{peak_lr:.1e} "
                f"({warmup_steps}/{total_train_steps} steps, ~{warmup_epochs_est:.1f} epochs), "
                f"后 {1 - args.warmup_ratio:.0%} 步 cosine→{min_lr:.1e} "
                f"({total_train_steps - warmup_steps} steps, ~{decay_epochs_est:.1f} epochs)",
                flush=True,
            )
    loss_fn = nn.SmoothL1Loss(reduction="none")

    model.train()

    start_epoch = 1
    step = 0
    loss_history: list[float] = []
    log_rows: list[tuple[int, int, int, float, float, float]] = []

    if args.resume:
        resume_path = _resolve_resume_path(args.resume, args.save_path)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume 找不到 checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if ckpt.get("format") != "pretrain_train_ckpt_v1":
            raise ValueError(
                f"checkpoint 格式不支持 resume: {resume_path}（需 pretrain_train_ckpt_v1，请用 --load_checkpoint 仅加载权重）"
            )
        _load_weights_into_model(model, ckpt)
        opt.load_state_dict(ckpt["optimizer"])
        if lr_scheduler is not None and ckpt.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        loss_history = list(ckpt.get("loss_history", []))
        log_rows = [tuple(row) for row in ckpt.get("log_rows", [])]
        if is_main:
            print(
                f"已从 checkpoint 恢复: {resume_path} "
                f"(下一 epoch={start_epoch}, step={step}, 历史 loss 点数={len(loss_history)})",
                flush=True,
            )
        if start_epoch > int(args.epochs):
            if is_main:
                print(f"checkpoint 已完成全部 {args.epochs} 个 epoch，无需继续训练。", flush=True)
            _dist_cleanup()
            return
    elif args.load_checkpoint:
        load_path = os.path.abspath(args.load_checkpoint)
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"--load_checkpoint 找不到文件: {load_path}")
        payload = torch.load(load_path, map_location=device, weights_only=False)
        _load_weights_into_model(model, payload)
        if is_main:
            print(f"已加载权重（从头训练）: {load_path}", flush=True)

    if is_main and args.checkpoint_every:
        print(f"训练 checkpoint: 每 {args.checkpoint_every} 个 epoch 保存（可 --resume latest）", flush=True)

    pbar_total = int(args.steps) if int(args.steps) > 0 else None
    pbar = tqdm(
        total=pbar_total,
        initial=step if pbar_total else 0,
        desc="pretrain",
        dynamic_ncols=True,
        disable=not is_main,
    )
    nan_skip_count = 0

    try:
        random.seed(args.csv_seed)
        for epoch in range(start_epoch, int(args.epochs) + 1):
            file_order = list(range(len(csv_files)))
            random.shuffle(file_order)

            # 方案 A：所有 rank 每 epoch 严格跑相同的 steps_per_epoch 步。
            # IterableDataset 各 rank 实际产出的 batch 数会有 ~±1% 偏差，
            # 如果让先耗尽的 rank 提前退出循环，后续 rank 的手写 all_reduce
            # （_dist_should_skip_step 等）会因找不到 peer 而 NCCL timeout。
            # 这里改成"固定 steps_per_epoch + dataloader 提前耗尽自动重启"，
            # 保证所有 rank 的 collective 调用次数一一对应，彻底消除发散。
            target_steps_in_epoch = max(int(steps_per_epoch), 1)
            epoch_step = 0
            cycle_idx = 0
            epoch_done = False

            while not epoch_done and epoch_step < target_steps_in_epoch:
                # 不同 cycle 用不同 rng 种子，避免循环时重复看到相同 shuffle
                effective_epoch = epoch * 1000 + cycle_idx
                current_dataloader = _build_epoch_dataloader(file_order, effective_epoch)
                cycle_idx += 1
                try:
                    for batch in current_dataloader:
                        file_i = int(batch["file_index"])
                        m = batch["marker_indices"].to(device, non_blocking=True)
                        v = batch["marker_values"].to(device, non_blocking=True)

                        mask = sample_mask(
                            batch_size=m.shape[0],
                            seq_len=m.shape[1],
                            mask_ratio=cfg.mask_ratio,
                            device=device,
                        )
                        pred_values, _ = model(m, v, mask=mask)
                        per_token = loss_fn(pred_values, v)
                        loss = per_token[mask].mean()

                        step_consumed = False
                        if _dist_should_skip_step(loss=loss, device=device):
                            if is_main:
                                nan_skip_count += 1
                                if nan_skip_count <= 10 or nan_skip_count % 200 == 0:
                                    print(
                                        f"  [skip] 非有限 loss，跳过本 step "
                                        f"(epoch={epoch}, file={file_i + 1}, 累计 skip={nan_skip_count})",
                                        flush=True,
                                    )
                            step_consumed = True
                        else:
                            opt.zero_grad(set_to_none=True)
                            loss.backward()
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                max_norm=float(args.max_grad_norm),
                            )
                            if _dist_should_skip_after_backward(model=model, device=device):
                                opt.zero_grad(set_to_none=True)
                                if is_main:
                                    nan_skip_count += 1
                                    if nan_skip_count <= 10 or nan_skip_count % 200 == 0:
                                        print(
                                            f"  [skip] 非有限梯度 (grad_norm={float(grad_norm):.4g})，跳过 optimizer.step "
                                            f"(epoch={epoch}, file={file_i + 1}, 累计 skip={nan_skip_count})",
                                            flush=True,
                                        )
                                step_consumed = True
                            else:
                                opt.step()
                                if lr_scheduler is not None:
                                    lr_scheduler.step()

                                loss_value = float(loss.item())
                                current_lr = opt.param_groups[0]["lr"]
                                if is_main:
                                    loss_history.append(loss_value)
                                    log_rows.append(
                                        (step, epoch, file_i + 1, loss_value, float(cfg.mask_ratio), current_lr)
                                    )

                                    if step % 20 == 0:
                                        try:
                                            pbar.set_postfix(
                                                loss=f"{loss_value:.4f}",
                                                lr=f"{current_lr:.2e}",
                                                mask_ratio=f"{cfg.mask_ratio:.3f}",
                                                cfg=str(asdict(cfg)),
                                                epoch=f"{epoch}/{args.epochs}",
                                                file=f"{file_i + 1}/{len(csv_files)}",
                                            )
                                        except Exception as e:
                                            print(f"  [warn] pbar.set_postfix 失败，忽略: {e}", flush=True)

                                    try:
                                        pbar.update(1)
                                    except Exception as e:
                                        print(f"  [warn] pbar.update 失败，忽略: {e}", flush=True)

                                step += 1
                                step_consumed = True

                        if step_consumed:
                            epoch_step += 1

                        if epoch_step >= target_steps_in_epoch:
                            epoch_done = True
                            break
                        if pbar_total is not None and step >= pbar_total:
                            epoch_done = True
                            break
                finally:
                    del current_dataloader

                if not epoch_done and is_main and epoch_step < target_steps_in_epoch:
                    print(
                        f"  [info] epoch {epoch}: dataloader 已耗尽于 {epoch_step}/{target_steps_in_epoch} "
                        f"步，重新构建 dataloader 继续（cycle={cycle_idx}）",
                        flush=True,
                    )

            if pbar_total is not None and step >= pbar_total:
                break

            # 所有 rank 都已跑满 target_steps_in_epoch，对齐后再存盘
            dist.barrier()
            if epoch % int(args.checkpoint_every) == 0:
                if is_main:
                    raw = _unwrap(model)
                    train_ckpt = _train_checkpoint_path(args.save_path, epoch)
                    latest_ckpt = _train_latest_checkpoint_path(args.save_path)
                    _save_training_checkpoint(
                        path=train_ckpt,
                        model=model,
                        opt=opt,
                        lr_scheduler=lr_scheduler,
                        cfg=cfg,
                        model_size=args.model_size,
                        epoch=epoch,
                        step=step,
                        loss_history=loss_history,
                        log_rows=log_rows,
                    )
                    _save_training_checkpoint(
                        path=latest_ckpt,
                        model=model,
                        opt=opt,
                        lr_scheduler=lr_scheduler,
                        cfg=cfg,
                        model_size=args.model_size,
                        epoch=epoch,
                        step=step,
                        loss_history=loss_history,
                        log_rows=log_rows,
                    )
                    enc_path = _encoder_epoch_checkpoint_path(args.save_path, epoch)
                    torch.save(
                        {"cfg": asdict(cfg), "state_dict": raw.encoder.state_dict()},
                        enc_path,
                    )
                    print(
                        f"已保存训练 checkpoint（可 --resume）: {train_ckpt}\n"
                        f"  latest: {latest_ckpt}\n"
                        f"  encoder: {enc_path}",
                        flush=True,
                    )
                dist.barrier()

        pbar.close()

        dist.barrier()

        save_root, save_ext = os.path.splitext(args.save_path)
        log_path = args.log_path if args.log_path else f"{save_root}_train_log.csv"
        curve_path = args.loss_curve_path if args.loss_curve_path else f"{save_root}_loss_curve.png"

        if is_main:
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "epoch", "file_index", "loss", "mask_ratio", "lr"])
                writer.writerows(log_rows)
            print(f"已保存训练日志：{log_path}")

            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(8, 4.5))
                plt.plot(range(len(loss_history)), loss_history, linewidth=1.2)
                plt.title("Pretrain Loss Curve")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(curve_path, dpi=150)
                plt.close()
                print(f"已保存损失曲线：{curve_path}")
            except Exception as e:
                print(f"绘制损失曲线失败：{e}")

            raw = _unwrap(model)
            payload = {"cfg": asdict(cfg), "state_dict": raw.encoder.state_dict()}
            torch.save(payload, args.save_path)
            print(f"已保存：{args.save_path}")

            _save_dir = os.path.dirname(os.path.abspath(args.save_path))
            full_ckpt_path = os.path.join(
                _save_dir or os.getcwd(),
                pretrain_mae_full_ckpt_name(args.model_size),
            )
            torch.save({"cfg": asdict(cfg), "state_dict": raw.state_dict()}, full_ckpt_path)
            print(f"已保存完整 MAE 权重（decoder+重建头，用于单 CSV 重建演示）：{full_ckpt_path}")

        dist.barrier()

    finally:
        _dist_cleanup()


if __name__ == "__main__":
    main()
