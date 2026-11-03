from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from common.patient_paper import (
    patient_labels,
    patient_table,
    predict_proba_cnn,
    save_predictions,
    split_cnn_rf_patients,
    train_histogram_cnn,
    write_json,
)


from patient_level.baselines.ensemble_cnn.model import build_images, pair_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-style EnsembleCNN: per marker-pair CNN logits aggregated by RF.")
    parser.add_argument("--split_xlsx", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_events", type=int, default=12000)
    parser.add_argument("--max_pairs", type=int, default=0, help="0 means all marker pairs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_rows = patient_table(args.split_xlsx, "train")
    test_rows = patient_table(args.split_xlsx, "test")
    labels = patient_labels(train_rows)
    cnn_ids, rf_ids = split_cnn_rf_patients(labels, args.seed)
    test_labels = patient_labels(test_rows)
    test_ids = test_labels["patient_id"].astype(str).tolist()

    encoder = LabelEncoder().fit(labels["label"].astype(str))
    y_lookup = dict(zip(labels["patient_id"].astype(str), encoder.transform(labels["label"].astype(str))))
    test_y_lookup = dict(zip(test_labels["patient_id"].astype(str), test_labels["label"].astype(str)))
    specs = pair_specs(train_rows, args.max_events, args.max_pairs)
    print({"device": args.device, "num_pairs": len(specs), "classes": encoder.classes_.tolist()}, flush=True)

    rf_train_features = []
    test_features = []
    cnn_dir = out / "pair_cnns"
    cnn_dir.mkdir(exist_ok=True)

    for idx, spec in enumerate(specs, start=1):
        print(f"[EnsembleCNN] pair {idx}/{len(specs)} {spec}", flush=True)
        x_cnn = build_images(train_rows, cnn_ids, spec, args.bins, args.max_events)
        y_cnn = np.asarray([y_lookup[pid] for pid in cnn_ids], dtype=np.int64)
        model = train_histogram_cnn(
            x_cnn,
            y_cnn,
            len(encoder.classes_),
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        torch.save({"spec": spec, "state_dict": model.state_dict(), "classes": encoder.classes_.tolist()}, cnn_dir / f"pair_{idx:04d}.pt")
        x_rf = build_images(train_rows, rf_ids, spec, args.bins, args.max_events)
        x_test = build_images(test_rows, test_ids, spec, args.bins, args.max_events)
        rf_train_features.append(predict_proba_cnn(model, x_rf, args.device))
        test_features.append(predict_proba_cnn(model, x_test, args.device))

    x_rf_all = np.hstack(rf_train_features)
    y_rf = np.asarray([y_lookup[pid] for pid in rf_ids], dtype=np.int64)
    x_test_all = np.hstack(test_features)

    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=args.seed)
    rf.fit(x_rf_all, y_rf)
    pred_idx = rf.predict(x_test_all)
    proba = rf.predict_proba(x_test_all)
    pred = encoder.inverse_transform(pred_idx)
    y_true = np.asarray([test_y_lookup[pid] for pid in test_ids])
    metrics = save_predictions(out, test_ids, y_true, pred, proba, encoder.classes_)

    joblib.dump({"rf": rf, "encoder": encoder, "specs": specs, "args": vars(args)}, out / "ensemble_cnn_paper.joblib")
    write_json(out / "metadata.json", {"method": "paper_ensemble_cnn", "args": vars(args), "num_pairs": len(specs), "metrics": metrics})
    print(metrics, flush=True)


if __name__ == "__main__":
    main()

