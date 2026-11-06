from __future__ import annotations

import os

MODEL_SIZE_NAMES: tuple[str, ...] = ("tiny", "small", "base")


def _check_model_size(model_size: str) -> str:
    if model_size not in MODEL_SIZE_NAMES:
        raise ValueError(f"model_size 必须是 {MODEL_SIZE_NAMES} 之一，当前为: {model_size}")
    return model_size


def pretrain_encoder_ckpt_name(model_size: str) -> str:
    """预训练 encoder 权重文件名，如 pretrained_tiny.pt"""
    return f"pretrained_{_check_model_size(model_size)}.pt"


def pretrain_mae_full_ckpt_name(model_size: str) -> str:
    """完整 MAE 权重文件名，如 pretrained_mae_full_small.pt"""
    return f"pretrained_mae_full_{_check_model_size(model_size)}.pt"


def finetuned_classifier_ckpt_name(model_size: str) -> str:
    """微调分类器权重文件名，如 finetuned_classifier_base.pt"""
    return f"finetuned_classifier_{_check_model_size(model_size)}.pt"


def finetune_train_log_name(model_size: str) -> str:
    return f"finetune_{_check_model_size(model_size)}_train_log.csv"


def finetune_loss_curve_name(model_size: str) -> str:
    return f"finetune_{_check_model_size(model_size)}_loss_curve.png"


def eval_metrics_name(model_size: str) -> str:
    return f"metrics_{_check_model_size(model_size)}.json"


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_experiment_layout(experiment_dir: str) -> dict[str, str]:
    root = os.path.abspath(experiment_dir)
    pretrain = os.path.join(root, "pretrain")
    finetune = os.path.join(root, "finetune")
    eval_dir = os.path.join(root, "eval")
    os.makedirs(pretrain, exist_ok=True)
    os.makedirs(finetune, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    return {
        "root": root,
        "pretrain": pretrain,
        "finetune": finetune,
        "eval": eval_dir,
    }
