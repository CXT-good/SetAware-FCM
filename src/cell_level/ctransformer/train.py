"""
阶段一训练脚本：静态细胞分类 Transformer
基于 FlowCytometryDataset 和 Stage1Transformer，在按管划分的 CSV 上训练单细胞分类器。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import math
from datetime import timedelta
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataio.paths import remap_manifest_path

from dataio.flow_dataset import FlowCytometryDataset
from cell_level.ctransformer.model import Stage1Transformer
from dataio.class_lists import dataset_key_from_split_xlsx
from cell_level.ctransformer.compute_epochs import load_steps_cache
from dataio.marker_mapping import (
    MARKER_TO_INDEX,
    LABEL_COLUMN,
    MARKER_LIST,
    MARKER_INDEX_LIST,
    CLASS_LIST,
    CLASS_NUMBER,
    load_class_list_for_dataset,
    default_class_mapping_path,
)

# 与 MAE train_pretrain.py 的 MODEL_SIZE_PRESETS 对齐（d_model/encoder/n_heads/d_ff）
STAGE1_MODEL_NAME = "stage1_transformer"
DEFAULT_CHECKPOINT_ROOT = "checkpoints"
CHECKPOINT_LATEST_NAME = "checkpoint_latest.pt"
CHECKPOINT_BEST_NAME = "checkpoint_best.pt"
INIT_MODES: tuple[str, ...] = ("random", "from_pretrain")

MODEL_SIZE_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {
        "embed_dim": 128,
        "num_layers": 6,
        "num_heads": 8,
        "dim_feedforward": 512,
    },
    "small": {
        "embed_dim": 384,
        "num_layers": 8,
        "num_heads": 16,
        "dim_feedforward": 1536,
    },
    "base": {
        "embed_dim": 512,
        "num_layers": 12,
        "num_heads": 16,
        "dim_feedforward": 2048,
    },
}


def seed_everything(seed: int) -> None:
    """Seed the Python, NumPy, and PyTorch RNGs used during training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _apply_model_size_preset(args: argparse.Namespace) -> None:
    for key, value in MODEL_SIZE_PRESETS[args.model_size].items():
        setattr(args, key, value)


def _dist_setup_optional() -> tuple[int, int, int, bool]:
    """若由 torchrun 启动且 WORLD_SIZE>=2，则初始化 DDP。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        return 0, 1, 0, False
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    # HDD 上各文件加载耗时差异大，集合通信超时放宽到 30 分钟，避免误杀。
    timeout_sec = int(os.environ.get("NCCL_COLLECTIVE_TIMEOUT_SEC", "1800"))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_sec))
    return rank, world_size, local_rank, True


def _dist_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _load_model_state_dict(model: nn.Module, state_dict: dict) -> None:
    _unwrap(model).load_state_dict(state_dict)


def _model_state_dict(model: nn.Module) -> dict:
    return _unwrap(model).state_dict()


def _ddp_all_ranks_ok(local_ok: bool, device: torch.device) -> bool:
    """Return True only if every DDP rank reports local_ok=True."""
    if not (dist.is_available() and dist.is_initialized()):
        return bool(local_ok)
    flag = torch.tensor(1 if local_ok else 0, device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _fast_forward_scheduler(
    scheduler: optim.lr_scheduler.LRScheduler, completed_epochs: int, steps_per_epoch: int
) -> None:
    for _ in range(max(0, int(completed_epochs) * int(steps_per_epoch))):
        scheduler.step()


def resolve_output_dir(
    *,
    checkpoint_root: str,
    train_xlsx: str,
    model_name: str,
    model_size: str,
    init_mode: str,
    output_dir_override: str = "",
) -> str:
    """
    输出目录：checkpoint_root / 数据集 / 模型 / model_size / init_mode
    - 数据集：由 train_xlsx 推导（*_dataset_split.xlsx -> dataset_key）
    - 模型：stage1_transformer（可用 --model_name 覆盖）
    - model_size：tiny / small / base
    - init_mode：random / from_pretrain
    """
    if output_dir_override:
        return os.path.abspath(output_dir_override)
    if init_mode not in INIT_MODES:
        raise ValueError(f"init_mode 必须是 {INIT_MODES} 之一，当前为: {init_mode}")
    dataset_key = dataset_key_from_split_xlsx(train_xlsx)
    root = os.path.abspath(checkpoint_root)
    return os.path.join(root, dataset_key, model_name, model_size, init_mode)


def _default_pretrained_path(checkpoint_root: str, model_size: str) -> str:
    return os.path.join(
        os.path.abspath(checkpoint_root),
        "pretrain",
        f"pretrained_{model_size}.pt",
    )


def _remap_mae_encoder_to_stage1(mae_state_dict: dict) -> dict:
    """将 MAE 预训练 encoder 的参数名映射到 Stage1Transformer 的命名。

    MAE encoder  ->  Stage1Transformer
      embed.marker_emb.*  ->  marker_embedding.*
      embed.value_emb.*   ->  value_embedding.*
      embed.value_ln.*    ->  value_ln.*
      enc.layers.*        ->  transformer.layers.*
    其余（embed.mask_token / 解码器等）跳过；cls_token / classifier 保持随机初始化。
    """
    prefix_map = (
        ("embed.marker_emb.", "marker_embedding."),
        ("embed.value_emb.", "value_embedding."),
        ("embed.value_ln.", "value_ln."),
        ("enc.layers.", "transformer.layers."),
    )
    remapped: dict = {}
    for key, value in mae_state_dict.items():
        new_key = None
        for src, dst in prefix_map:
            if key.startswith(src):
                new_key = dst + key[len(src):]
                break
        if new_key is not None:
            remapped[new_key] = value
    return remapped


def _load_pretrained_encoder(model: nn.Module, pretrained_path: str) -> None:
    """加载 MAE 预训练 encoder 权重到 Stage1Transformer（分类头随机初始化）。"""
    if not pretrained_path or not os.path.isfile(pretrained_path):
        raise FileNotFoundError(f"预训练权重不存在: {pretrained_path}")
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    mae_sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    remapped = _remap_mae_encoder_to_stage1(mae_sd)
    model_sd = model.state_dict()
    loadable = {
        k: v
        for k, v in remapped.items()
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
    }
    skipped = [k for k in remapped if k not in loadable]
    result = model.load_state_dict(loadable, strict=False)
    kept_random = [k for k in result.missing_keys]
    print(
        f"已加载预训练 encoder 参数 {len(loadable)} 个；"
        f"跳过(形状/名称不匹配) {len(skipped)} 个；"
        f"随机初始化(分类头等) {len(kept_random)} 个: {kept_random}"
    )
    if not loadable:
        raise RuntimeError(
            f"未能从 {pretrained_path} 加载任何 encoder 参数，请检查权重文件与模型结构是否匹配。"
        )


def _save_training_metrics_plot(
    output_dir: str,
    epoch_losses: list[float],
    epoch_f1s: list[float],
    *,
    best_epoch: int = 0,
) -> str:
    """保存训练 loss / f1 曲线（按实际完成的 epoch 数绘图）。"""
    if not epoch_losses:
        return ""
    n_epochs = len(epoch_losses)
    epochs_range = range(1, n_epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, epoch_losses, label="avg_loss", marker="o")
    plt.plot(epochs_range, epoch_f1s, label="avg_f1", marker="s")
    if best_epoch > 0:
        plt.axvline(
            x=best_epoch,
            color="green",
            linestyle="--",
            alpha=0.6,
            label=f"best epoch {best_epoch}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Training Metrics Trend")
    plt.legend()
    plt.grid(alpha=0.3)
    plot_path = os.path.join(output_dir, "training_metrics.pdf")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def _write_run_meta(output_dir: str, meta: dict) -> None:
    path = os.path.join(output_dir, "run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _count_csv_rows_fast(csv_path: str) -> int:
    """快速估算 CSV 行数（去掉表头）。假设常规 CSV 不含跨行字段。"""
    with open(csv_path, "rb") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


def _steps_per_epoch_for_paths(
    train_csv_paths: list[str],
    batch_size: int,
    *,
    dataset_key: str = "",
    steps_cache_path: str = "",
) -> int:
    """统计 steps/epoch；优先用缓存，避免 HDD 上数百个 CSV 逐文件数行卡住。"""
    if dataset_key and steps_cache_path and os.path.isfile(steps_cache_path):
        steps_map = load_steps_cache(steps_cache_path)
        if dataset_key in steps_map:
            cached = int(steps_map[dataset_key])
            print(
                f"steps_per_epoch={cached}（来自 {steps_cache_path}，跳过逐 CSV 数行）",
                flush=True,
            )
            return cached

    n_files = len(train_csv_paths)
    print(
        f"正在统计 {n_files} 个训练 CSV 行数（HDD 上可能需数分钟，并非卡死）...",
        flush=True,
    )
    steps = 0
    for i, p in enumerate(train_csv_paths, 1):
        n = _count_csv_rows_fast(p)
        if n > 0:
            steps += int(math.ceil(n / float(batch_size)))
        if i == 1 or i % 20 == 0 or i == n_files:
            print(f"  数行进度 {i}/{n_files}  steps_per_epoch≈{steps}", flush=True)
    print(f"steps_per_epoch={steps}", flush=True)
    return steps


def _build_warmup_cosine_scheduler(
    *,
    model: nn.Module,
    train_csv_paths: list[str],
    epochs: int,
    batch_size: int,
    lr_start: float,
    lr_peak: float,
    lr_end: float,
    warmup_ratio: float,
    weight_decay: float,
    dataset_key: str = "",
    steps_cache_path: str = "",
    world_size: int = 1,
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler, dict]:
    """
    按 step 的 warmup + cosine 调度：
    - warmup: lr_start -> lr_peak，占总 step 的 warmup_ratio
    - cosine: lr_peak -> lr_end
    """
    global_steps_per_epoch = _steps_per_epoch_for_paths(
        train_csv_paths,
        batch_size,
        dataset_key=dataset_key,
        steps_cache_path=steps_cache_path,
    )
    # DDP stage1 keeps every rank on the same file/batch schedule to avoid Join/custom
    # all_reduce ordering conflicts. Because ranks no longer shard files, each rank runs
    # global_steps_per_epoch optimizer steps per epoch.
    steps_per_epoch = global_steps_per_epoch
    total_steps = max(1, int(steps_per_epoch) * int(epochs))
    warmup_steps = int(round(total_steps * float(warmup_ratio)))
    warmup_steps = max(1, min(warmup_steps, total_steps))
    cosine_steps = max(1, total_steps - warmup_steps)

    optimizer = optim.Adam(model.parameters(), lr=float(lr_peak), weight_decay=weight_decay)

    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(lr_start) / float(lr_peak),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=float(lr_end),
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
    meta = {
        "steps_per_epoch": int(steps_per_epoch),
        "global_steps_per_epoch": int(global_steps_per_epoch),
        "total_steps": int(total_steps),
        "warmup_steps": int(warmup_steps),
        "cosine_steps": int(cosine_steps),
    }
    return optimizer, scheduler, meta


def train_epoch(
    model: nn.Module,
    train_csv_paths: list[str],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
    args: argparse.Namespace,
    max_grad_norm: float = 0.5,
    class_to_index: Optional[dict[str, int]] = None,
    *,
    rank: int = 0,
    world_size: int = 1,
    is_main: bool = True,
    epoch_index: int = 0,
) -> tuple[float, float]:
    """训练一个 epoch，并打印每个批次的指标。"""
    model.train()
    epoch_loss = 0.0
    epoch_f1 = 0.0
    total_samples = 0

    # 每个 epoch 打乱 CSV 文件顺序。DDP 下所有 rank 使用相同文件顺序并各自遍历
    # 全部 CSV，确保每一步 collective 顺序完全一致。不要和 Join 混用：本循环中
    # 还有自定义 all_reduce finite checks，Join 的影子 allreduce 会打乱 collective 顺序。
    file_paths = list(train_csv_paths)
    rng = random.Random(int(args.seed) + int(epoch_index))
    rng.shuffle(file_paths)

    for csv_path in file_paths:
        # 逐文件构建 dataset/dataloader，避免一次性把所有 CSV 缓存进内存
        dataset = FlowCytometryDataset(
            data_path=csv_path,
            marker_columns=None,  # 自动读取除 label/Time/event_ID 外的所有列
            label_column=LABEL_COLUMN,
            class_to_index=class_to_index,
        )
        use_cuda = device.type == "cuda"
        dl_kwargs: dict = {
            "batch_size": args.batch_size,
            "shuffle": True,
            "num_workers": args.num_workers,
            "pin_memory": use_cuda,
        }
        if args.num_workers > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = 2
        dataloader = DataLoader(dataset, **dl_kwargs)

        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                desc=f"Training {os.path.basename(csv_path)}",
                disable=not is_main,
            ),
            start=1,
        ):
            non_blocking = use_cuda
            marker_indices = batch["marker_indices"][0].to(device, non_blocking=non_blocking)
            marker_values = batch["marker_values"].to(device, non_blocking=non_blocking)
            labels = batch["label"].to(device, non_blocking=non_blocking)

            optimizer.zero_grad()
            logits, _ = model(marker_values, marker_indices)
            target = labels.view(-1)

            # 防御性检查：NaN / Inf。DDP 下必须让所有 rank 做同一个跳过决定，
            # 否则一个 rank continue、另一个 rank backward 会卡在 collective。
            local_ok = bool(torch.isfinite(logits).all().item())
            if not local_ok and is_main:
                print(f"警告: Batch {batch_idx} logits 包含 NaN 或 Inf，所有 rank 同步跳过此批次")

            loss = None
            if local_ok:
                loss = criterion(logits, target)
                local_ok = bool(torch.isfinite(loss).all().item())
                if not local_ok and is_main:
                    print(f"警告: Batch {batch_idx} loss 为 NaN 或 Inf，所有 rank 同步跳过此批次")

            if not _ddp_all_ranks_ok(local_ok, device):
                optimizer.zero_grad()
                continue

            assert loss is not None
            loss.backward()

            has_nan_inf_grad = any(
                p.grad is not None
                and (not torch.isfinite(p.grad).all())
                for p in model.parameters()
            )
            if not _ddp_all_ranks_ok(not has_nan_inf_grad, device):
                if is_main:
                    print(f"警告: Batch {batch_idx} 梯度包含 NaN 或 Inf，所有 rank 同步跳过此批次")
                optimizer.zero_grad()
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm
            )
            grad_norm_ok = bool(torch.isfinite(grad_norm).all().item())
            if not _ddp_all_ranks_ok(grad_norm_ok, device):
                if is_main:
                    print(f"警告: Batch {batch_idx} 梯度范数异常，所有 rank 同步跳过此批次")
                optimizer.zero_grad()
                continue

            optimizer.step()
            scheduler.step()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            batch_labels = labels.view(-1).cpu().numpy()
            # macro：各类别指标等权平均，不受样本量影响（适合类别不均衡）
            f1 = f1_score(batch_labels, preds, average="macro", zero_division=0)
            precision = precision_score(
                batch_labels, preds, average="macro", zero_division=0
            )
            recall = recall_score(
                batch_labels, preds, average="macro", zero_division=0
            )

            batch_size = labels.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_f1 += f1 * batch_size
            total_samples += batch_size

            if is_main:
                print(
                    f"Batch {batch_idx}: "
                    f"avg_loss={loss.item():.4f}, "
                    f"f1={f1:.4f}, "
                    f"precision={precision:.4f}, "
                    f"recall={recall:.4f}"
                )

    if world_size > 1 and dist.is_initialized():
        stats = torch.tensor(
            [epoch_loss, epoch_f1, float(total_samples)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        epoch_loss = float(stats[0].item())
        epoch_f1 = float(stats[1].item())
        total_samples = int(stats[2].item())

    avg_loss = epoch_loss / total_samples if total_samples > 0 else 0.0
    avg_f1 = epoch_f1 / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="训练阶段一 Transformer 模型")
    parser.add_argument(
        "--train_xlsx",
        type=str,
        required=True,
        help="训练集 split xlsx 的绝对路径（将从 train 子表读取 csv_path）",
    )
    parser.add_argument(
        "--xlsx_csv_col",
        type=str,
        default="csv_path",
        help="xlsx 中 CSV 路径列名（默认 csv_path）",
    )
    parser.add_argument(
        "--class_mapping_path",
        type=str,
        default=default_class_mapping_path(),
        help="逐细胞类别映射 JSON（由 build_class_mappings.py 生成）",
    )
    parser.add_argument(
        "--steps_cache",
        type=str,
        default="cache/dataset_train_steps_cache.json",
        help="steps/epoch 缓存；有则跳过启动时逐 CSV 数行",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="checkpoint 根目录；实际输出为 <root>/<数据集>/<模型>/<model_size>/<init_mode>/",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=STAGE1_MODEL_NAME,
        help="模型名称目录（默认 stage1_transformer）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="覆盖自动输出路径；默认按 checkpoint_root/数据集/模型/model_size/init_mode 生成",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="tiny",
        choices=sorted(MODEL_SIZE_PRESETS),
        help="模型规模预设：tiny（默认）/ small / base；与 MAE 预训练一致",
    )
    parser.add_argument(
        "--init_mode",
        type=str,
        default="random",
        choices=list(INIT_MODES),
        help="权重初始化：random=随机初始化；from_pretrain=加载 MAE encoder",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="",
        help="MAE encoder 权重路径；init_mode=from_pretrain 时使用，默认 checkpoint/pretrain/pretrained_<model_size>.pt",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    parser.add_argument("--seed", type=int, default=42, help="训练随机种子")
    parser.add_argument("--batch_size", type=int, default=4096, help="批次大小")
    parser.add_argument("--epochs", type=int, default=60, help="训练轮数")
    # 注意：当前训练使用 warmup+cosine 调度，基础学习率由 --lr_peak 控制；
    # --lr 参数保留兼容性但不再用于优化器实际 lr。
    parser.add_argument("--lr", type=float, default=1e-4, help="（兼容保留，不使用）")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--lr_start", type=float, default=5e-5, help="学习率起始值（warmup 起点）")
    parser.add_argument("--lr_peak", type=float, default=1e-4, help="学习率峰值（warmup 终点/余弦起点）")
    parser.add_argument("--lr_end", type=float, default=1e-5, help="学习率结束值（余弦终点）")
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="warmup 占总 epoch 的比例（默认 0.1，即前 10%%）",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=len(CLASS_LIST),
        help="分类类别数（会自动从数据中检测，此参数仅用于未找到标签列时的默认值）",
    )
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="是否保存嵌入向量（当前脚本未使用，仅保留接口）",
    )
    parser.add_argument(
        "--device", type=int, default=None, help="指定 GPU 编号（如 0 或 1），不传则自动选择"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="数据加载器使用的进程数"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.5, help="梯度裁剪最大范数"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="每隔多少个 epoch 覆盖保存 checkpoint_latest.pt（训练结束必存一次）",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="early stopping：连续多少个 epoch 的 avg_loss 无改善则提前停止（<=0 关闭）",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="判定 avg_loss 改善的最小降幅（默认 0，即严格更低才算改善）",
    )
    parser.add_argument(
        "--value_embedding_mode",
        type=str,
        default="shared",
        choices=["shared"],
        help="数值投影方式：shared=所有 marker 共享 Linear(1->embed_dim)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 output_dir 中的 checkpoint_latest.pt 断点续训",
    )

    args = parser.parse_args()
    _apply_model_size_preset(args)
    seed_everything(args.seed)

    rank, world_size, local_rank, distributed = _dist_setup_optional()
    is_main = rank == 0

    # 设备
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    elif args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_main:
        print(f"使用设备: {device}" + (f" (DDP world_size={world_size})" if distributed else ""))
        print(
            f"模型规模: {args.model_size}  "
            f"(embed_dim={args.embed_dim}, num_layers={args.num_layers}, "
            f"num_heads={args.num_heads}, dim_feedforward={args.dim_feedforward})"
        )

    if not os.path.isabs(args.train_xlsx):
        raise ValueError(f"--train_xlsx 必须是绝对路径: {args.train_xlsx}")
    if not os.path.isfile(args.train_xlsx):
        raise FileNotFoundError(f"--train_xlsx 文件不存在: {args.train_xlsx}")

    if args.init_mode == "from_pretrain" and not args.pretrained_path:
        args.pretrained_path = _default_pretrained_path(
            args.checkpoint_root, args.model_size
        )
    args.pretrained_path = (
        os.path.abspath(args.pretrained_path) if args.pretrained_path else ""
    )

    args.output_dir = resolve_output_dir(
        checkpoint_root=args.checkpoint_root,
        train_xlsx=args.train_xlsx,
        model_name=args.model_name,
        model_size=args.model_size,
        init_mode=args.init_mode,
        output_dir_override=args.output_dir,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    for old_ckpt in glob.glob(os.path.join(args.output_dir, "checkpoint_epoch_*.pt")):
        try:
            os.remove(old_ckpt)
        except OSError:
            pass
    print(
        f"输出目录: {args.output_dir}  "
        f"(数据集={dataset_key_from_split_xlsx(args.train_xlsx)}, "
        f"模型={args.model_name}, size={args.model_size}, init={args.init_mode})"
    )
    if args.init_mode == "from_pretrain":
        if is_main:
            print(f"预训练权重: {args.pretrained_path}")
    if is_main:
        print(f"batch_size={args.batch_size}")

    # 从训练 xlsx（train 子表）收集 CSV 列表，并从 class_mapping 加载逐细胞类别映射

    class_mapping_path = args.class_mapping_path
    if not os.path.isabs(class_mapping_path):
        class_mapping_path = os.path.abspath(class_mapping_path)
    if not os.path.isfile(class_mapping_path):
        raise FileNotFoundError(f"class_mapping 不存在: {class_mapping_path}")

    datasets_spec: list[str] = []
    class_to_index: dict[str, int] = {}
    class_counts_for_weight: list[int] = []
    # 固定从 train 子表读取
    df = pd.read_excel(args.train_xlsx, sheet_name="train")
    if args.xlsx_csv_col not in df.columns:
        raise ValueError(
            f"xlsx 缺少 CSV 路径列 {args.xlsx_csv_col}，当前列: {list(df.columns)}"
        )
    raw_paths = df[args.xlsx_csv_col].astype(str).tolist()

    ds_key = dataset_key_from_split_xlsx(args.train_xlsx)
    class_list, class_counts = load_class_list_for_dataset(
        ds_key, mapping_path=class_mapping_path
    )
    class_to_index = {c: i for i, c in enumerate(class_list)}
    class_counts_for_weight = [int(x) for x in class_counts]
    if is_main:
        print(f"dataset_key={ds_key}")
        print(f"class_mapping: n_classes={len(class_list)}")

    missing = 0
    for p in raw_paths:
        p = os.path.abspath(remap_manifest_path(str(p).strip()))
        if not p.lower().endswith(".csv"):
            continue
        if not os.path.isfile(p):
            missing += 1
            continue
        datasets_spec.append(p)
    if missing and is_main:
        print(f"警告: xlsx 中有 {missing} 个 csv_path 不存在（已跳过）")

    if not datasets_spec:
        raise ValueError(
            "train 子表中没有可用的 CSV 路径，请检查 xlsx 的 csv_path 列及文件是否存在"
        )

    if is_main:
        print(f"将加载 CSV 文件数量: {len(datasets_spec)}")
        if class_to_index:
            print(f"xlsx 类别映射: {class_to_index}")

    # 注意：不在这里构建 DataLoader 列表，避免一次性加载所有 CSV 到内存；
    # 训练时在 train_epoch 中逐文件构建并迭代。

    # 创建模型
    if class_to_index:
        args.num_classes = len(class_to_index)
    model = Stage1Transformer(
        markers=MARKER_INDEX_LIST,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_classes=args.num_classes,
        value_embedding_mode=args.value_embedding_mode,
    ).to(device)

    if args.init_mode == "from_pretrain":
        _load_pretrained_encoder(model, args.pretrained_path)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    if is_main:
        print(f"模型参数数量: {sum(p.numel() for p in _unwrap(model).parameters()):,}")

    # 类别权重
    if class_counts_for_weight and len(class_counts_for_weight) == args.num_classes:
        class_counts = [int(count) for count in class_counts_for_weight]
        total_samples = sum(class_counts)
        class_weights = torch.tensor(
            [
                total_samples / (args.num_classes * count) if count > 0 else 1.0
                for count in class_counts
            ],
            dtype=torch.float32,
            device=device,
        )
        class_weights = class_weights / class_weights.mean()
        if is_main:
            print(f"类别样本数量: {class_counts}")
            print(f"类别权重: {class_weights.cpu().numpy()}")
    elif len(CLASS_NUMBER) == args.num_classes and len(CLASS_NUMBER) > 0:
        class_counts = [int(count) for count in CLASS_NUMBER]
        total_samples = sum(class_counts)
        class_weights = torch.tensor(
            [
                total_samples / (args.num_classes * count) if count > 0 else 1.0
                for count in class_counts
            ],
            dtype=torch.float32,
            device=device,
        )
        class_weights = class_weights / class_weights.mean()
        if is_main:
            print(f"类别样本数量: {class_counts}")
            print(f"类别权重: {class_weights.cpu().numpy()}")
    else:
        if is_main:
            print(
                f"警告: CLASS_NUMBER长度({len(CLASS_NUMBER)})与类别数({args.num_classes})不匹配，使用平衡权重"
            )
        class_weights = torch.ones(
            args.num_classes, dtype=torch.float32, device=device
        )
        if is_main:
            print(f"类别权重: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    lr_start = float(args.lr_start)
    lr_peak = float(args.lr_peak)
    lr_end = float(args.lr_end)
    steps_cache = (
        os.path.abspath(args.steps_cache) if args.steps_cache else ""
    )
    optimizer, scheduler, sched_meta = _build_warmup_cosine_scheduler(
        model=model,
        train_csv_paths=datasets_spec,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_start=lr_start,
        lr_peak=lr_peak,
        lr_end=lr_end,
        warmup_ratio=float(args.warmup_ratio),
        weight_decay=args.weight_decay,
        dataset_key=ds_key,
        steps_cache_path=steps_cache,
        world_size=world_size,
    )
    if is_main:
        print(
            f"lr_schedule(step): start={lr_start:g}, peak={lr_peak:g}, end={lr_end:g}, "
            f"warmup_steps={sched_meta['warmup_steps']}/{sched_meta['total_steps']} (ratio={args.warmup_ratio:g}), "
            f"steps_per_epoch≈{sched_meta['steps_per_epoch']}"
            + (
                f" (global≈{sched_meta['global_steps_per_epoch']}, world_size={world_size})"
                if world_size > 1
                else ""
            )
        )

    if is_main:
        print("\n开始训练...")
    epoch_losses: list[float] = []
    epoch_f1s: list[float] = []
    start_epoch = 0

    best_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    resume_path = os.path.join(args.output_dir, CHECKPOINT_LATEST_NAME)
    if args.resume:
        best_path = os.path.join(args.output_dir, CHECKPOINT_BEST_NAME)
        losses_path = os.path.join(args.output_dir, "epoch_losses.npy")
        if os.path.isfile(resume_path):
            pass
        elif os.path.isfile(best_path) and os.path.isfile(losses_path):
            resume_path = best_path
            if is_main:
                print(
                    f"未找到 {CHECKPOINT_LATEST_NAME}，从 {CHECKPOINT_BEST_NAME} 续训"
                )
        else:
            raise FileNotFoundError(
                f"--resume 已指定，但断点不存在: {resume_path} 或 {best_path}"
            )
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        ckpt_epoch = int(ckpt.get("epoch", 0))
        losses_path = os.path.join(args.output_dir, "epoch_losses.npy")
        f1s_path = os.path.join(args.output_dir, "epoch_f1s.npy")
        if os.path.isfile(losses_path):
            epoch_losses = [float(x) for x in np.load(losses_path).tolist()]
        if os.path.isfile(f1s_path):
            epoch_f1s = [float(x) for x in np.load(f1s_path).tolist()]
        completed_epochs = len(epoch_losses)
        if completed_epochs > ckpt_epoch and os.path.isfile(best_path):
            best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
            if int(best_ckpt.get("epoch", 0)) >= completed_epochs:
                ckpt = best_ckpt
                ckpt_epoch = int(best_ckpt.get("epoch", 0))
                resume_path = best_path
                if is_main:
                    print(
                        f"checkpoint_latest epoch={ckpt_epoch} 落后于 epoch_losses="
                        f"{completed_epochs}，改用 {CHECKPOINT_BEST_NAME} 续训"
                    )
        start_epoch = max(ckpt_epoch, completed_epochs)
        _load_model_state_dict(model, ckpt["model_state_dict"])
        ckpt_world = int(ckpt.get("world_size", 1))
        if ckpt_world == world_size and ckpt_epoch >= completed_epochs:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            if is_main and ckpt_world != world_size:
                print(
                    f"警告: 断点 world_size={ckpt_world} 与当前 {world_size} 不同，"
                    f"仅恢复模型权重并重同步 LR schedule"
                )
            elif is_main and ckpt_epoch < completed_epochs:
                print(
                    f"断点 optimizer 与 epoch_losses 不完全对齐，"
                    f"重同步 LR schedule 到 step {start_epoch * sched_meta['steps_per_epoch']}"
                )
            _fast_forward_scheduler(
                scheduler, start_epoch, sched_meta["steps_per_epoch"]
            )
        if len(epoch_losses) > start_epoch:
            if is_main:
                print(
                    f"断点 epoch={start_epoch}，epoch_losses 有 {len(epoch_losses)} 条，"
                    f"截断后从 Epoch {start_epoch + 1} 重训"
                )
            epoch_losses = epoch_losses[:start_epoch]
            epoch_f1s = epoch_f1s[:start_epoch]
        if epoch_losses:
            best_epoch = int(np.argmin(epoch_losses)) + 1
            best_loss = float(epoch_losses[best_epoch - 1])
            epochs_no_improve = len(epoch_losses) - best_epoch
        if is_main:
            print(
                f"已从断点恢复: {resume_path} (已完成 {start_epoch} epoch，"
                f"下一 epoch={start_epoch + 1}/{args.epochs}，"
                f"best epoch {best_epoch}, best_loss={best_loss:.4f})"
            )

    def _build_checkpoint(epoch_num: int) -> dict:
        return {
            "epoch": epoch_num,
            "world_size": world_size,
            "model_state_dict": _model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "num_markers": len(MARKER_INDEX_LIST),
            "num_classes": args.num_classes,
            "marker_to_idx": MARKER_TO_INDEX,
            "dataset_key": ds_key,
            "class_list": class_list,
            "class_to_index": class_to_index,
            "model_name": args.model_name,
            "model_size": args.model_size,
            "init_mode": args.init_mode,
            "pretrained_path": args.pretrained_path or None,
            "checkpoint_root": os.path.abspath(args.checkpoint_root),
            "output_dir": args.output_dir,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "value_embedding_mode": args.value_embedding_mode,
        }

    try:
        for epoch in range(start_epoch, args.epochs):
            if is_main:
                print(f"\nEpoch {epoch + 1}/{args.epochs}")
            avg_loss, avg_f1 = train_epoch(
                model,
                datasets_spec,
                criterion,
                optimizer,
                scheduler,
                device,
                args,
                max_grad_norm=args.max_grad_norm,
                class_to_index=class_to_index if class_to_index else None,
                rank=rank,
                world_size=world_size,
                is_main=is_main,
                epoch_index=epoch,
            )
            if is_main:
                print(
                    f"Epoch {epoch + 1} summary: avg_loss={avg_loss:.4f}, avg_f1={avg_f1:.4f}"
                )
            epoch_losses.append(avg_loss)
            epoch_f1s.append(avg_f1)

            if is_main:
                np.save(os.path.join(args.output_dir, "epoch_losses.npy"), epoch_losses)
                np.save(os.path.join(args.output_dir, "epoch_f1s.npy"), epoch_f1s)

            # 按 avg_loss 低谷保存最优权重（checkpoint_best.pt）
            improved = avg_loss < (best_loss - float(args.min_delta))
            if improved:
                best_loss = avg_loss
                best_epoch = epoch + 1
                epochs_no_improve = 0
                if is_main:
                    best_path = os.path.join(args.output_dir, CHECKPOINT_BEST_NAME)
                    torch.save(_build_checkpoint(epoch + 1), best_path)
                    print(
                        f"已保存最优权重: {best_path} (epoch {best_epoch}, avg_loss={best_loss:.4f})"
                    )
            else:
                epochs_no_improve += 1

            # 仅保留一个最新权重文件（覆盖写入 checkpoint_latest.pt）
            save_interval = max(1, int(args.save_interval))
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == args.epochs:
                if is_main:
                    checkpoint_path = os.path.join(args.output_dir, CHECKPOINT_LATEST_NAME)
                    torch.save(_build_checkpoint(epoch + 1), checkpoint_path)
                    print(f"已保存权重: {checkpoint_path} (epoch {epoch + 1})")

            # early stopping：连续 patience 个 epoch 无改善则提前停止
            if int(args.patience) > 0 and epochs_no_improve >= int(args.patience):
                if is_main:
                    print(
                        f"\n[early stopping] 连续 {epochs_no_improve} 个 epoch avg_loss 无改善"
                        f"（最优 epoch {best_epoch}, avg_loss={best_loss:.4f}），在 epoch {epoch + 1} 提前停止。"
                    )
                    torch.save(
                        _build_checkpoint(epoch + 1),
                        os.path.join(args.output_dir, CHECKPOINT_LATEST_NAME),
                    )
                break

        if is_main:
            print(f"\n训练结束：最优权重在 epoch {best_epoch} (avg_loss={best_loss:.4f})")

            # 保存 epoch 曲线
            np.save(os.path.join(args.output_dir, "epoch_losses.npy"), epoch_losses)
            np.save(os.path.join(args.output_dir, "epoch_f1s.npy"), epoch_f1s)
            print(f"epoch_losses 和 epoch_f1s 已保存至: {args.output_dir}")

            _write_run_meta(
                args.output_dir,
                {
                    "model_name": args.model_name,
                    "model_size": args.model_size,
                    "init_mode": args.init_mode,
                    "pretrained_path": args.pretrained_path or None,
                    "train_xlsx": os.path.abspath(args.train_xlsx),
                    "dataset_key": ds_key,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr_start": lr_start,
                    "lr_peak": lr_peak,
                    "lr_end": lr_end,
                    "warmup_ratio": float(args.warmup_ratio),
                    "output_dir": args.output_dir,
                    "world_size": world_size,
                },
            )
            print(f"已保存 run_meta.json: {os.path.join(args.output_dir, 'run_meta.json')}")

            if epoch_losses:
                try:
                    plot_path = _save_training_metrics_plot(
                        args.output_dir,
                        epoch_losses,
                        epoch_f1s,
                        best_epoch=best_epoch,
                    )
                    print(f"训练曲线已保存至: {plot_path}")
                except Exception as e:
                    print(f"警告: 训练曲线保存失败（不影响 checkpoint）: {e}")

            print("\n训练完成！")
    finally:
        _dist_cleanup()


if __name__ == "__main__":
    main()
