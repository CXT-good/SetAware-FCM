"""
基线模型训练与评估：*_dataset_split.xlsx（train / test 子表）
- 逻辑回归、DeepCyTOF：标记物交集
- XGBoost：标记物并集（缺失填 0）
"""
from __future__ import annotations

import argparse
import json
import os

import math

from dataio.baseline_loaders import count_train_samples_for_split, load_split_xlsx
from dataio.class_lists import dataset_key_from_split_xlsx
from dataio.marker_mapping import default_class_mapping_path
from cell_level.baselines.deepcytof.model import train_eval_deepcytof
from cell_level.baselines.logistic_regression.model import train_eval_logistic_regression
from cell_level.baselines.xgboost.model import train_eval_xgboost
from cell_level.ctransformer.compute_epochs import (
    REF_DATASET_KEY,
    REF_EPOCHS,
    compute_epochs_from_sample_count,
)

DEEPCYTOF_BATCH_SIZE = 4096


def _default_class_mapping() -> str:
    return default_class_mapping_path()


def _build_parser(method: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="split xlsx 基线模型训练（在 test 子表上评估）"
    )
    parser.add_argument(
        "--dataset_xlsx",
        required=True,
        help="*_dataset_split.xlsx 绝对路径；训练读 train 子表，评估读 test 子表（同一文件）",
    )
    parser.add_argument(
        "--class_mapping_path",
        default=_default_class_mapping(),
        help="datasets_class_mappings.json 路径",
    )
    parser.add_argument(
        "--result_root",
        default="results",
        help="结果根目录：{result_root}/{dataset}/<method>/",
    )
    parser.add_argument(
        "--max_cells_per_file",
        type=int,
        default=50000,
        help="训练集每文件最大采样细胞数；0 表示不限制。测试集始终全量。",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[method] if method is not None else ["lr", "deepcytof", "xgboost"],
        choices=[method] if method is not None else ["lr", "deepcytof", "xgboost"],
    )
    parser.add_argument(
        "--deepcytof_epochs",
        type=int,
        default=0,
        help="DeepCyTOF epoch 数；0 表示按 Zhongnan 60 epoch 总迭代次数自动对齐（batch_size=4096）",
    )
    parser.add_argument(
        "--epochs_cache",
        default="cache/dataset_train_steps_cache.json",
        help="compute_epochs 使用的 steps 缓存",
    )
    parser.add_argument(
        "--max_deepcytof_epochs",
        type=int,
        default=0,
        help="DeepCyTOF epoch 上限；0 表示不限制",
    )
    return parser


def main(method: str | None = None) -> None:
    args = _build_parser(method).parse_args()

    dataset_xlsx = os.path.abspath(args.dataset_xlsx)
    if not os.path.isabs(dataset_xlsx):
        raise SystemExit("--dataset_xlsx 必须是绝对路径")

    ds_key = dataset_key_from_split_xlsx(dataset_xlsx)
    out_root = os.path.join(args.result_root, ds_key)
    os.makedirs(out_root, exist_ok=True)

    max_train = None if args.max_cells_per_file <= 0 else args.max_cells_per_file
    load_kw = dict(
        dataset_xlsx=dataset_xlsx,
        class_mapping_path=os.path.abspath(args.class_mapping_path),
        max_cells_per_file_train=max_train,
        max_cells_per_file_test=None,
    )

    results: dict[str, float] = {}
    meta_base = {
        "dataset_key": ds_key,
        "dataset_xlsx": dataset_xlsx,
        "class_mapping_path": os.path.abspath(args.class_mapping_path),
        "max_cells_per_file_train": args.max_cells_per_file,
    }

    if "lr" in args.methods:
        print("\n" + "=" * 60)
        print("逻辑回归 (交集)")
        print("=" * 60)
        X_tr, y_tr, X_te, y_te, markers, class_list, class_to_index = load_split_xlsx(
            **load_kw,
            mode="intersection",
        )
        print(f"训练集: {X_tr.shape}, 测试集: {X_te.shape}")
        lr_dir = os.path.join(out_root, "logistic_regression")
        results["logistic_regression"] = train_eval_logistic_regression(
            X_tr, y_tr, X_te, y_te, lr_dir, class_list=class_list
        )
        with open(os.path.join(lr_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {**meta_base, "mode": "intersection", "n_markers": len(markers), "class_to_index": class_to_index},
                f,
                indent=2,
                ensure_ascii=False,
            )

    if "deepcytof" in args.methods:
        X_tr, y_tr, X_te, y_te, markers, class_list, class_to_index = load_split_xlsx(
            **load_kw,
            mode="intersection",
        )
        n_train = int(X_tr.shape[0])
        data_dir = os.path.dirname(dataset_xlsx)
        ref_xlsx = os.path.join(data_dir, f"{REF_DATASET_KEY}_dataset_split.xlsx")
        if ds_key == REF_DATASET_KEY:
            n_ref_train = n_train
        else:
            n_ref_train = count_train_samples_for_split(
                ref_xlsx,
                max_cells_per_file=max_train,
            )

        deep_epochs = args.deepcytof_epochs
        if deep_epochs <= 0:
            deep_epochs = compute_epochs_from_sample_count(
                n_train,
                n_ref_train,
                ref_epochs=REF_EPOCHS,
                batch_size=DEEPCYTOF_BATCH_SIZE,
            )
        if args.max_deepcytof_epochs > 0:
            deep_epochs = min(deep_epochs, args.max_deepcytof_epochs)

        steps_per_epoch = max(1, math.ceil(n_train / DEEPCYTOF_BATCH_SIZE))
        ref_steps_per_epoch = max(1, math.ceil(n_ref_train / DEEPCYTOF_BATCH_SIZE))
        target_total_steps = REF_EPOCHS * ref_steps_per_epoch
        print("\n" + "=" * 60)
        print(f"DeepCyTOF (交集) epochs={deep_epochs}")
        print(
            f"迭代对齐: 基准={REF_DATASET_KEY} {REF_EPOCHS} epoch "
            f"(~{target_total_steps} steps, batch={DEEPCYTOF_BATCH_SIZE})；"
            f"本集 train={n_train}, steps/epoch={steps_per_epoch}, "
            f"总 steps~{deep_epochs * steps_per_epoch}"
        )
        print("=" * 60)
        print(f"训练集: {X_tr.shape}, 测试集: {X_te.shape}")
        dc_dir = os.path.join(out_root, "deepcytof")
        results["deepcytof"] = train_eval_deepcytof(
            X_tr,
            y_tr,
            X_te,
            y_te,
            dc_dir,
            class_list=class_list,
            epochs=deep_epochs,
            batch_size=DEEPCYTOF_BATCH_SIZE,
        )
        with open(os.path.join(dc_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    **meta_base,
                    "mode": "intersection",
                    "n_markers": len(markers),
                    "class_to_index": class_to_index,
                    "deepcytof_epochs": deep_epochs,
                    "deepcytof_batch_size": DEEPCYTOF_BATCH_SIZE,
                    "ref_dataset_key": REF_DATASET_KEY,
                    "ref_epochs": REF_EPOCHS,
                    "n_train_samples": n_train,
                    "n_ref_train_samples": n_ref_train,
                    "steps_per_epoch": steps_per_epoch,
                    "target_total_steps": target_total_steps,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    if "xgboost" in args.methods:
        print("\n" + "=" * 60)
        print("XGBoost (并集)")
        print("=" * 60)
        X_tr, y_tr, X_te, y_te, markers, class_list, class_to_index = load_split_xlsx(
            **load_kw,
            mode="union",
        )
        print(f"训练集: {X_tr.shape}, 测试集: {X_te.shape}")
        xgb_dir = os.path.join(out_root, "xgboost")
        results["xgboost"] = train_eval_xgboost(
            X_tr, y_tr, X_te, y_te, xgb_dir, class_list=class_list
        )
        with open(os.path.join(xgb_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {**meta_base, "mode": "union", "n_markers": len(markers), "class_to_index": class_to_index},
                f,
                indent=2,
                ensure_ascii=False,
            )

    summary_path = os.path.join(out_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 60)
    print(f"数据集 {ds_key} 汇总 weighted F1:")
    for name, f1 in results.items():
        print(f"  {name}: {f1:.4f}")
    print(f"结果目录: {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
