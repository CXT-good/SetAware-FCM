from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from dataio.baseline_loaders import count_train_samples_for_split
from dataio.class_lists import dataset_key_from_split_xlsx
from dataio.marker_mapping import load_class_list_for_dataset
from cell_level.baselines.deepcytof.model import DeepCyTOFMLP
from cell_level.multicenter.common import (
    CENTER_XLSX,
    COMMON_MARKERS,
    iter_csv_chunks,
    load_source_training_data,
    read_split_paths,
    write_metric_artifacts,
)
from cell_level.ctransformer.compute_epochs import (
    REF_DATASET_KEY,
    REF_EPOCHS,
    compute_epochs_from_sample_count,
)


METHODS = ("logistic_regression", "xgboost", "deepcytof")


def xgboost_params(num_classes: int, seed: int) -> dict:
    return {
        "objective": "multi:softmax",
        "num_class": int(num_classes),
        "max_depth": 8,
        "eta": 0.1,
        "tree_method": "hist",
        "device": "cuda",
        "seed": int(seed),
    }


def train_traditional_arrays(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    class_list: Sequence[str],
    output_dir: str | os.PathLike[str],
    *,
    marker_columns: Sequence[str] = COMMON_MARKERS,
    seed: int = 42,
    max_iter: int = 500,
    xgb_rounds: int = 200,
    deepcytof_epochs: int = 50,
    batch_size: int = 4096,
    device: str = "cuda",
) -> None:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if x_train.shape[1] != len(marker_columns):
        raise ValueError("training feature count does not match marker_columns")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler().fit(x_train)
    x_scaled = scaler.transform(x_train).astype(np.float32)

    if method == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            max_iter=max_iter, C=1.0, solver="lbfgs", random_state=seed
        ).fit(x_scaled, y_train)
        np.savez(
            out / "model.npz",
            coef=model.coef_,
            intercept=model.intercept_,
            classes=model.classes_,
            scale_mean=scaler.mean_,
            scale_std=scaler.scale_,
            class_list=np.asarray(class_list, dtype=object),
            marker_columns=np.asarray(marker_columns, dtype=object),
        )
    elif method == "xgboost":
        import xgboost as xgb

        dtrain = xgb.DMatrix(x_scaled, label=y_train)
        booster = xgb.train(
            xgboost_params(len(class_list), seed),
            dtrain,
            num_boost_round=xgb_rounds,
            verbose_eval=False,
        )
        booster.save_model(out / "model.json")
        np.savez(
            out / "scaler.npz",
            mean=scaler.mean_,
            std=scaler.scale_,
            class_list=np.asarray(class_list, dtype=object),
            marker_columns=np.asarray(marker_columns, dtype=object),
        )
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = DeepCyTOFMLP(len(marker_columns), len(class_list)).to(torch_device)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x_scaled), torch.from_numpy(y_train).long()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(deepcytof_epochs):
            model.train()
            loss_sum = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(torch_device), yb.to(torch_device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item())
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(
                    f"DeepCyTOF epoch={epoch + 1}/{deepcytof_epochs} "
                    f"loss={loss_sum / max(1, len(loader)):.6f}",
                    flush=True,
                )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_std": scaler.scale_,
                "n_features": len(marker_columns),
                "n_classes": len(class_list),
                "class_list": list(class_list),
                "marker_columns": list(marker_columns),
            },
            out / "model.pt",
        )


def _load_class_list(split_xlsx: str, class_mapping_path: str) -> list[str]:
    dataset_key = dataset_key_from_split_xlsx(split_xlsx)
    class_list, _ = load_class_list_for_dataset(
        dataset_key, mapping_path=class_mapping_path
    )
    return list(class_list)


def _predictor(method: str, model_dir: Path, device: str):
    if method == "logistic_regression":
        artifact = np.load(model_dir / "model.npz", allow_pickle=True)
        mean, scale = artifact["scale_mean"], artifact["scale_std"]
        coef, intercept = artifact["coef"], artifact["intercept"]
        classes = artifact["classes"].astype(np.int64)

        def predict(x):
            scores = ((x - mean) / np.where(scale == 0, 1, scale)) @ coef.T + intercept
            if scores.shape[1] == 1 and len(classes) == 2:
                return classes[(scores[:, 0] > 0).astype(np.int64)]
            return classes[np.argmax(scores, axis=1)]

        return predict, artifact["class_list"].tolist(), artifact["marker_columns"].tolist()
    if method == "xgboost":
        import xgboost as xgb

        artifact = np.load(model_dir / "scaler.npz", allow_pickle=True)
        booster = xgb.Booster()
        booster.load_model(model_dir / "model.json")
        mean, scale = artifact["mean"], artifact["std"]

        def predict(x):
            scaled = (x - mean) / np.where(scale == 0, 1, scale)
            return booster.predict(xgb.DMatrix(scaled)).astype(np.int64)

        return predict, artifact["class_list"].tolist(), artifact["marker_columns"].tolist()

    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = DeepCyTOFMLP(checkpoint["n_features"], checkpoint["n_classes"]).to(torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = torch.from_numpy(checkpoint["scaler_mean"]).float().to(torch_device)
    scale = torch.from_numpy(checkpoint["scaler_std"]).float().to(torch_device)

    def predict(x):
        xb = torch.from_numpy(x).float().to(torch_device)
        with torch.no_grad():
            logits = model((xb - mean) / torch.where(scale == 0, 1, scale))
        return logits.argmax(dim=1).cpu().numpy()

    return predict, checkpoint["class_list"], checkpoint["marker_columns"]


def evaluate_traditional(
    method: str,
    model_dir: str,
    test_xlsx: str,
    output_dir: str,
    *,
    train_center: str,
    test_center: str,
    chunksize: int = 200_000,
    device: str = "cuda",
) -> None:
    predict, class_list, markers = _predictor(method, Path(model_dir), device)
    if tuple(markers) != COMMON_MARKERS:
        raise ValueError(f"artifact marker order is not the common panel: {markers}")
    class_to_index = {str(name): index for index, name in enumerate(class_list)}
    confusion = np.zeros((len(class_list), len(class_list)), dtype=np.int64)
    samples = 0
    for path in read_split_paths(test_xlsx, "test"):
        for x, y in iter_csv_chunks(
            path, markers, class_to_index, chunksize=chunksize
        ):
            prediction = predict(x)
            confusion += confusion_matrix(
                y, prediction, labels=np.arange(len(class_list))
            )
            samples += len(y)
        print(f"evaluated {Path(path).name}", flush=True)
    write_metric_artifacts(
        confusion,
        class_list,
        output_dir,
        {
            "method": method,
            "train_center": train_center,
            "test_center": test_center,
            "model_dir": str(Path(model_dir).resolve()),
            "test_xlsx": str(Path(test_xlsx).resolve()),
            "marker_columns": markers,
            "n_samples": samples,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict cross-center traditional baselines")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--method", choices=METHODS, required=True)
    train.add_argument("--center", choices=CENTER_XLSX, required=True)
    train.add_argument("--split_xlsx", default="")
    train.add_argument("--class_mapping_path", required=True)
    train.add_argument("--output_dir", required=True)
    train.add_argument("--max_cells_per_file", type=int, default=50_000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max_iter", type=int, default=500)
    train.add_argument("--xgb_rounds", type=int, default=200)
    train.add_argument("--deepcytof_epochs", type=int, default=0)
    train.add_argument("--batch_size", type=int, default=4096)
    train.add_argument("--device", default="cuda")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--method", choices=METHODS, required=True)
    evaluate.add_argument("--model_dir", required=True)
    evaluate.add_argument("--train_center", choices=CENTER_XLSX, required=True)
    evaluate.add_argument("--test_center", choices=CENTER_XLSX, required=True)
    evaluate.add_argument("--test_xlsx", default="")
    evaluate.add_argument("--output_dir", required=True)
    evaluate.add_argument("--chunksize", type=int, default=200_000)
    evaluate.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "evaluate":
        evaluate_traditional(
            args.method,
            args.model_dir,
            args.test_xlsx or CENTER_XLSX[args.test_center],
            args.output_dir,
            train_center=args.train_center,
            test_center=args.test_center,
            chunksize=args.chunksize,
            device=args.device,
        )
        return

    split_xlsx = args.split_xlsx or CENTER_XLSX[args.center]
    class_list = _load_class_list(split_xlsx, args.class_mapping_path)
    x_train, y_train, paths = load_source_training_data(
        split_xlsx,
        class_list,
        max_cells_per_file=args.max_cells_per_file,
        seed=args.seed,
    )
    deep_epochs = args.deepcytof_epochs
    if args.method == "deepcytof" and deep_epochs <= 0:
        reference = str(Path(split_xlsx).with_name(f"{REF_DATASET_KEY}_dataset_split.xlsx"))
        n_reference = count_train_samples_for_split(
            reference, max_cells_per_file=args.max_cells_per_file
        )
        deep_epochs = compute_epochs_from_sample_count(
            len(y_train),
            n_reference,
            ref_epochs=REF_EPOCHS,
            batch_size=args.batch_size,
        )
    train_traditional_arrays(
        args.method,
        x_train,
        y_train,
        class_list,
        args.output_dir,
        marker_columns=COMMON_MARKERS,
        seed=args.seed,
        max_iter=args.max_iter,
        xgb_rounds=args.xgb_rounds,
        deepcytof_epochs=deep_epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    (Path(args.output_dir) / "run_meta.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "train_center": args.center,
                "split_xlsx": str(Path(split_xlsx).resolve()),
                "marker_columns": list(COMMON_MARKERS),
                "class_list": class_list,
                "train_csvs": paths,
                "n_train_samples": len(y_train),
                "seed": args.seed,
                "deepcytof_epochs": deep_epochs if args.method == "deepcytof" else None,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
