from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from dataio.baseline_loaders import load_split_xlsx, save_metrics_per_class
from dataio.class_lists import dataset_key_from_split_xlsx, macro_label_indices
from dataio.marker_mapping import default_class_mapping_path

DEFAULT_OUTPUT_ROOT = "outputs"
METHOD_SUBDIRS = {
    "lr": "logistic_regression",
    "xgboost": "xgboost",
    "deepcytof": "deepcytof",
}


def _measure_forward_only_torch(
    forward_fn,
    x: "torch.Tensor",
    warmup_batches: int,
    batch_size: int,
    device: "torch.device",
) -> dict:
    import torch

    n = int(x.shape[0])
    timed_batches = 0
    timed_samples = 0
    forward_seconds_total = 0.0

    with torch.no_grad():
        for i in range(warmup_batches):
            s = i * batch_size
            if s >= n:
                break
            xb = x[s : min(n, s + batch_size)]
            _ = forward_fn(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for s in range(0, n, batch_size):
            xb = x[s : min(n, s + batch_size)]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = forward_fn(xb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timed_batches += 1
            timed_samples += int(xb.shape[0])
            forward_seconds_total += float(t1 - t0)

    return {
        "timed_batches": timed_batches,
        "timed_samples": timed_samples,
        "forward_seconds_total": forward_seconds_total,
        "forward_seconds_per_sample": (
            forward_seconds_total / timed_samples if timed_samples else None
        ),
    }


def _predict_batched_torch(
    x: "torch.Tensor",
    predict_batch: Callable[["torch.Tensor"], np.ndarray],
    batch_size: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    n = int(x.shape[0])
    for s in range(0, n, batch_size):
        xb = x[s : min(n, s + batch_size)]
        parts.append(predict_batch(xb))
    return np.concatenate(parts)


def _save_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_list: List[str],
    out_dir: Path,
    *,
    dataset_key: str | None = None,
) -> tuple[float, float]:
    n_classes = len(class_list)
    labels = list(range(n_classes))
    supports = np.bincount(y_true.astype(int), minlength=n_classes).tolist()
    macro_labels = macro_label_indices(class_list, dataset_key, supports=supports)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(
        y_true, y_pred, average="macro", labels=macro_labels, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_list, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=class_list, columns=class_list).to_csv(
        out_dir / "confusion_matrix.csv"
    )
    save_metrics_per_class(
        cm,
        str(out_dir / "metrics_per_class.csv"),
        class_names=class_list,
        dataset_key=dataset_key,
    )
    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"weighted_f1: {f1_w:.4f}\nmacro_f1: {f1_m:.4f}\n\n")
        f.write(report)
    print(f"每类指标已保存至: {out_dir / 'metrics_per_class.csv'}")
    return f1_w, f1_m


def _default_class_mapping() -> str:
    return default_class_mapping_path()


def resolve_baseline_eval_output_dir(
    *,
    output_root: str,
    dataset_xlsx: str,
    method: str,
    output_dir_override: str = "",
) -> str:
    """评估输出：output_root / 数据集 / method_subdir（与 checkpoint 布局一致）。"""
    if output_dir_override:
        return os.path.abspath(output_dir_override)
    ds_key = dataset_key_from_split_xlsx(dataset_xlsx)
    subdir = METHOD_SUBDIRS[method]
    return os.path.join(os.path.abspath(output_root), ds_key, subdir)


def _build_parser(method: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="基线模型在 test 子表上评估（每类指标 + 推理计时）"
    )
    p.add_argument(
        "--method",
        choices=[method] if method is not None else ["lr", "xgboost", "deepcytof"],
        default=method,
        required=method is None,
    )
    p.add_argument(
        "--model_dir",
        required=True,
        help="checkpoint 中模型权重目录（只读，含 model.npz / model.json / model.pt）",
    )
    p.add_argument(
        "--dataset_xlsx",
        required=True,
        help="*_dataset_split.xlsx 绝对路径（与训练一致；测试数据来自 test 子表）",
    )
    p.add_argument(
        "--output_root",
        default=DEFAULT_OUTPUT_ROOT,
        help="评估结果根目录；输出为 <root>/<数据集>/<method>/",
    )
    p.add_argument(
        "--output_dir",
        default="",
        help="覆盖自动评估输出路径",
    )
    p.add_argument(
        "--class_mapping_path",
        default=_default_class_mapping(),
    )
    p.add_argument("--device", default="cuda:0", help="e.g. cuda:0 or cpu")
    p.add_argument("--batch_size", type=int, default=65536)
    p.add_argument("--warmup_batches", type=int, default=5)
    p.add_argument("--timing_output", default="inference_timing.json")
    return p


def main(method: str | None = None) -> None:
    args = _build_parser(method).parse_args()

    dataset_xlsx = os.path.abspath(args.dataset_xlsx)
    ds_key = dataset_key_from_split_xlsx(dataset_xlsx)
    model_dir = Path(args.model_dir)
    out_dir = Path(
        resolve_baseline_eval_output_dir(
            output_root=args.output_root,
            dataset_xlsx=dataset_xlsx,
            method=args.method,
            output_dir_override=args.output_dir,
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, X_test, y_test, _, class_list, _ = load_split_xlsx(
        dataset_xlsx,
        "union" if args.method == "xgboost" else "intersection",
        class_mapping_path=os.path.abspath(args.class_mapping_path),
        max_cells_per_file_train=None,
        max_cells_per_file_test=None,
    )

    model_name = METHOD_SUBDIRS[args.method]

    if args.method == "lr":
        import torch

        npz = model_dir / "model.npz"
        if not npz.exists():
            npz = model_dir / "model_outputs" / "model.npz"
        data = np.load(npz, allow_pickle=True)
        classes = data["classes"]
        coef = torch.from_numpy(data["coef"]).float()
        intercept = torch.from_numpy(data["intercept"]).float()
        mean = torch.from_numpy(data["scale_mean"]).float()
        std = torch.from_numpy(data["scale_std"]).float()

        device = torch.device(args.device)
        X = torch.from_numpy(X_test).float()
        Xs = (X - mean) / std
        Xs = Xs.to(device)
        coef = coef.to(device)
        intercept = intercept.to(device)

        def forward_fn(xb: torch.Tensor):
            return xb @ coef.t() + intercept

        def predict_batch(xb: torch.Tensor) -> np.ndarray:
            with torch.no_grad():
                scores = forward_fn(xb)
                if scores.shape[1] == 1:
                    # sklearn's binary decision function selects classes_[1]
                    # only for a strictly positive score; zero selects classes_[0].
                    indices = (scores[:, 0] > 0).long()
                else:
                    indices = scores.argmax(dim=1)
                return classes[indices.cpu().numpy()]

        y_pred = _predict_batched_torch(Xs, predict_batch, args.batch_size)
        timing_core = _measure_forward_only_torch(
            forward_fn=forward_fn,
            x=Xs,
            warmup_batches=args.warmup_batches,
            batch_size=args.batch_size,
            device=device,
        )
    elif args.method == "xgboost":
        import torch

        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("请安装 xgboost: pip install xgboost") from e

        model_json = model_dir / "model.json"
        scaler_npz = model_dir / "scaler.npz"
        if not model_json.exists():
            raise FileNotFoundError(f"未找到 XGBoost 模型: {model_json}")
        if not scaler_npz.exists():
            raise FileNotFoundError(f"未找到 scaler: {scaler_npz}")

        scaler = np.load(scaler_npz, allow_pickle=True)
        mean = torch.from_numpy(scaler["mean"]).float()
        std = torch.from_numpy(scaler["std"]).float()
        bst = xgb.Booster()
        bst.load_model(str(model_json))

        device = torch.device(args.device)
        X = torch.from_numpy(X_test).float()
        Xs = (X - mean) / std
        Xs = Xs.to(device)

        def forward_fn(xb: torch.Tensor):
            dmat = xgb.DMatrix(xb.detach().cpu().numpy())
            return bst.predict(dmat)

        def predict_batch(xb: torch.Tensor) -> np.ndarray:
            return forward_fn(xb).astype(np.int64)

        y_pred = _predict_batched_torch(Xs, predict_batch, args.batch_size)
        timing_core = _measure_forward_only_torch(
            forward_fn=forward_fn,
            x=Xs,
            warmup_batches=args.warmup_batches,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        import torch
        import torch.nn as nn

        pt = model_dir / "model.pt"
        if not pt.exists():
            pt = model_dir / "model_outputs" / "model.pt"
        ckpt = torch.load(pt, map_location="cpu", weights_only=False)
        mean = ckpt["scaler_mean"]
        std = ckpt["scaler_std"]
        n_features = int(ckpt["n_features"])
        n_classes = int(ckpt["n_classes"])

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_features, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, n_classes),
                )

            def forward(self, x):
                return self.net(x)

        device = torch.device(args.device)
        model = MLP().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        X = torch.from_numpy(X_test).float()
        Xs = (X - torch.from_numpy(mean).float()) / torch.from_numpy(std).float()
        Xs = Xs.to(device)

        def forward_fn(xb: torch.Tensor):
            return model(xb)

        def predict_batch(xb: torch.Tensor) -> np.ndarray:
            with torch.no_grad():
                return model(xb).argmax(dim=1).cpu().numpy()

        y_pred = _predict_batched_torch(Xs, predict_batch, args.batch_size)
        timing_core = _measure_forward_only_torch(
            forward_fn=forward_fn,
            x=Xs,
            warmup_batches=args.warmup_batches,
            batch_size=args.batch_size,
            device=device,
        )

    f1_w, f1_m = _save_classification_metrics(
        y_test, y_pred, class_list, out_dir, dataset_key=ds_key
    )
    print(f"{model_name} weighted F1: {f1_w:.4f}  macro F1: {f1_m:.4f}")

    result = {
        "task": "inference_forward_only",
        "model": model_name,
        "model_dir": str(model_dir.resolve()),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "warmup_batches": int(args.warmup_batches),
        "test_samples": int(X_test.shape[0]),
        "weighted_f1": float(f1_w),
        "macro_f1": float(f1_m),
        **timing_core,
    }
    out_path = out_dir / args.timing_output
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
