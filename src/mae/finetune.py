from __future__ import annotations

import argparse
import csv
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.mae_dataset import LabeledFlowCytometryDataset
from mae.layout import (
    MODEL_SIZE_NAMES,
    ensure_experiment_layout,
    ensure_parent_dir,
    finetune_loss_curve_name,
    finetune_train_log_name,
    finetuned_classifier_ckpt_name,
    pretrain_encoder_ckpt_name,
)
from dataio.marker_mapping import LABEL_COL, MARKER_LIST, CLASS_NUMBER
from mae.model import CellClassifier, MAEConfig, MAEEncoder

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=None, help="指定 GPU 编号（如 0 或 1），不传则自动选择（不要传负数）")
    p.add_argument("--epochs", type=int, default=1, help="训练轮数（Epoch 主导）")
    p.add_argument("--steps", type=int, default=500, help="仅用于日志中的 step 计数，不作为停止条件")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_classes", type=int, required=True, help="细胞类别数量（必填）")
    p.add_argument("--experiment_dir", type=str, default="", help="与预训练相同的实验根目录；读 pretrain/pretrained_<model_size>.pt，写 finetune/finetuned_classifier_<model_size>.pt。")
    p.add_argument(
        "--model_size",
        type=str,
        default="tiny",
        choices=list(MODEL_SIZE_NAMES),
        help="须与预训练 --model_size 一致（用于权重文件名中的 tiny/small/base）",
    )
    p.add_argument("--pretrained_path", type=str, default="", help="预训练 encoder 权重；不设时默认为 pretrained_<model_size>.pt。")
    p.add_argument("--fold_dir", type=str, default="", help="数据折目录：未指定 --train_dir 时训练 CSV 目录为 <fold_dir>/train。")
    p.add_argument("--train_dir", type=str, default="", help="微调训练集：含标签的 CSV 所在目录。优先于 --data_dir；可与 --fold_dir 联用（只指定 fold_dir 时用 fold_dir/train）。")
    p.add_argument("--data_dir", type=str, default="", help="带标签 CSV 目录（与 --train_dir 二选一；新流程建议用 --train_dir 或 --fold_dir）。")
    p.add_argument("--label_col", type=str, default="", help="标签列名（不传则使用 marker_mapping.LABEL_COL）。")
    p.add_argument("--csv_recursive", action="store_true", help="递归搜索 data_dir 下所有 CSV")
    p.add_argument("--csv_max_files", type=int, default=0, help="限制最多读取多少个 CSV（0 表示不限制）")
    p.add_argument("--csv_seed", type=int, default=0)
    p.add_argument("--log_path", type=str, default="", help="训练日志 CSV；不设时 finetune_<model_size>_train_log.csv。")
    p.add_argument("--loss_curve_path", type=str, default="", help="loss 曲线图；不设时 finetune_<model_size>_loss_curve.png。")
    p.add_argument("--save_path", type=str, default="", help="微调分类器权重；不设时 finetuned_classifier_<model_size>.pt。")
    args = p.parse_args()

    if args.experiment_dir:
        layout = ensure_experiment_layout(args.experiment_dir)
        if not args.pretrained_path:
            args.pretrained_path = os.path.join(
                layout["pretrain"], pretrain_encoder_ckpt_name(args.model_size)
            )
        if not args.save_path:
            args.save_path = os.path.join(
                layout["finetune"], finetuned_classifier_ckpt_name(args.model_size)
            )
        if not args.log_path:
            args.log_path = os.path.join(layout["finetune"], finetune_train_log_name(args.model_size))
        if not args.loss_curve_path:
            args.loss_curve_path = os.path.join(
                layout["finetune"], finetune_loss_curve_name(args.model_size)
            )
    else:
        if not args.pretrained_path:
            args.pretrained_path = pretrain_encoder_ckpt_name(args.model_size)
        if not args.save_path:
            args.save_path = finetuned_classifier_ckpt_name(args.model_size)
        if not args.log_path:
            args.log_path = finetune_train_log_name(args.model_size)
        if not args.loss_curve_path:
            args.loss_curve_path = finetune_loss_curve_name(args.model_size)

    ensure_parent_dir(args.save_path)
    ensure_parent_dir(args.log_path)
    ensure_parent_dir(args.loss_curve_path)

    train_data_dir = args.train_dir.strip()
    if not train_data_dir and args.fold_dir.strip():
        train_data_dir = os.path.join(args.fold_dir.strip(), "train")
    if not train_data_dir:
        train_data_dir = args.data_dir.strip()

    if not train_data_dir:
        raise ValueError("请提供 --train_dir、或 --fold_dir（将使用 fold_dir/train）、或 --data_dir")

    # 设备
    if args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"微调训练 CSV 目录: {os.path.abspath(train_data_dir)}")

    # 载入预训练好的 student encoder（微调阶段不再需要 teacher）
    ckpt = torch.load(args.pretrained_path, map_location="cpu")
    cfg = MAEConfig(**ckpt["cfg"])
    encoder = MAEEncoder(cfg)
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.to(device)

    label_col = args.label_col.strip() if args.label_col else LABEL_COL
    if not label_col:
        raise ValueError("请提供 --label_col，或在 src/data2vec_flow/marker_mapping.py 中设置 LABEL_COL。")

    marker_list = list(MARKER_LIST)
    if len(marker_list) != cfg.seq_len:
        raise ValueError(f"marker_list 长度({len(marker_list)}) 必须等于预训练的 seq_len({cfg.seq_len})")

    def _list_csv_files(data_dir: str, *, recursive: bool, max_files: int) -> list[str]:
        out: list[str] = []
        if recursive:
            for root, _dirs, files in os.walk(data_dir):
                for f in files:
                    if f.lower().endswith(".csv"):
                        out.append(os.path.join(root, f))
                        if max_files and len(out) >= max_files:
                            return sorted(out)
        else:
            for f in os.listdir(data_dir):
                if f.lower().endswith(".csv"):
                    out.append(os.path.join(data_dir, f))
                    if max_files and len(out) >= max_files:
                        return sorted(out)
        return sorted(out)

    csv_files = _list_csv_files(train_data_dir, recursive=bool(args.csv_recursive), max_files=args.csv_max_files)
    if not csv_files:
        raise ValueError(f"在 {train_data_dir} 下未找到任何 CSV")

    def _open_loader_for_file(i: int) -> DataLoader:
        ds = LabeledFlowCytometryDataset(
            data_path=csv_files[i],
            marker_columns=marker_list,
            label_column=label_col,
            xuhao_column="",
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    dataloader_list = [_open_loader_for_file(i) for i in range(len(csv_files))]

    # 细胞级分类：encoder 输出做 mean-pool，再接线性层
    clf = CellClassifier(encoder, num_classes=args.num_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr)
    # 类别不平衡处理：模仿 FCM/train_stage2.py
    # class_weights = total/(num_classes*count)，并做归一化使平均权重=1。
    num_classes = int(args.num_classes)
    if len(CLASS_NUMBER) == num_classes and len(CLASS_NUMBER) > 0:
        class_counts = [int(c) for c in CLASS_NUMBER]
        total_samples = int(sum(class_counts))
        class_weights = torch.tensor(
            [
                (total_samples / (num_classes * count)) if count > 0 else 1.0
                for count in class_counts
            ],
            dtype=torch.float32,
            device=device,
        )
        class_weights = class_weights / class_weights.mean()
        print(f"类别样本数量: {class_counts}")
        print(f"类别权重（归一化后，mean=1）: {class_weights.detach().cpu().numpy()}")
    else:
        print(
            f"警告: CLASS_NUMBER长度({len(CLASS_NUMBER)})与 num_classes({num_classes})不匹配，"
            f"使用均衡权重(全1)。"
        )
        class_weights = torch.ones(num_classes, dtype=torch.float32, device=device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    clf.train()
    step = 0
    pbar = tqdm(total=args.steps, desc="finetune", dynamic_ncols=True)
    rng = random.Random(args.csv_seed)
    loss_history: list[float] = []
    log_rows: list[tuple[int, int, int, float, float]] = []
    for epoch in range(args.epochs):
        # 与 FCM 对齐：每个 epoch 打乱“按文件构建的 dataloader 列表”
        rng.shuffle(dataloader_list)
        for file_i, current_dataloader in enumerate(dataloader_list):
            for batch in current_dataloader:
                m = batch["marker_indices"].to(device)#标记ID
                v = batch["marker_values"].to(device)#值
                y = batch["label"].to(device).view(-1)#标签

                # 标准监督分类损失
                logits = clf(m, v)#分类器输出
                loss = loss_fn(logits, y)#损失函数
                loss_value = float(loss.item())
                acc = float((logits.argmax(dim=-1) == y).float().mean().item())

                opt.zero_grad(set_to_none=True)#梯度清零
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=1.0)#梯度裁剪
                opt.step()
                loss_history.append(loss_value)
                log_rows.append((step, epoch + 1, file_i + 1, loss_value, acc))

                if step % 20 == 0:
                    pbar.set_postfix(
                        loss=f"{loss_value:.4f}",
                        acc=f"{acc:.3f}",
                        epoch=f"{epoch + 1}/{args.epochs}",
                        file=f"{file_i + 1}/{len(csv_files)}",
                    )
                pbar.update(1)
                step += 1

        if (epoch + 1) % 5 == 0:
            save_root, save_ext = os.path.splitext(args.save_path)
            ckpt_path = f"{save_root}_epoch{epoch + 1}{save_ext or '.pt'}"
            payload = {
                "cfg": ckpt["cfg"],
                "num_classes": args.num_classes,
                "state_dict": clf.state_dict(),
            }
            torch.save(payload, ckpt_path)
            print(f"已保存 checkpoint：{ckpt_path}")

    pbar.close()
    with open(args.log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "file_index", "loss", "acc"])
        writer.writerows(log_rows)
    print(f"已保存训练日志：{args.log_path}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4.5))
        plt.plot(range(len(loss_history)), loss_history, linewidth=1.2)
        plt.title("Finetune Loss Curve")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.loss_curve_path, dpi=150)
        plt.close()
        print(f"已保存损失曲线：{args.loss_curve_path}")
    except Exception as e:
        print(f"绘制损失曲线失败：{e}")

    payload = {
        "cfg": ckpt["cfg"],
        "num_classes": args.num_classes,
        "state_dict": clf.state_dict(),
    }
    torch.save(payload, args.save_path)
    print(f"已保存：{args.save_path}")


if __name__ == "__main__":
    main()

