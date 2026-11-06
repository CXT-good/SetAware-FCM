from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from common.patient_paper import (
    patient_labels,
    patient_table,
    save_predictions,
    write_json,
)


from patient_level.baselines.umap_rf.model import (
    balanced_tube_events,
    feature_matrix,
    import_umap,
    tube_order,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-style UMAP-RF: per-panel UMAP, 32x32 histograms, RF classifier.")
    parser.add_argument("--split_xlsx", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--max_total_events_per_tube", type=int, default=200000)
    parser.add_argument("--max_events_per_file", type=int, default=3000)
    parser.add_argument("--max_events", type=int, default=12000)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    umap = import_umap()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_rows = patient_table(args.split_xlsx, "train")
    test_rows = patient_table(args.split_xlsx, "test")
    train_labels = patient_labels(train_rows)
    test_labels = patient_labels(test_rows)
    train_ids = train_labels["patient_id"].astype(str).tolist()
    test_ids = test_labels["patient_id"].astype(str).tolist()
    tubes = tube_order(train_rows)
    print({"tubes": tubes, "bins": args.bins, "max_total_events_per_tube": args.max_total_events_per_tube}, flush=True)

    reducers = {}
    widths = {}
    for tube in tubes:
        print(f"[UMAP-RF] fitting tube {tube}", flush=True)
        values, width = balanced_tube_events(train_rows, tube, args.max_total_events_per_tube, args.max_events_per_file, args.seed)
        reducer = umap.UMAP(n_components=2, n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=None, n_jobs=-1).fit(values)
        reducers[tube] = reducer
        widths[tube] = width
        joblib.dump({"reducer": reducer, "width": width}, out / f"umap_tube_{tube}.joblib")

    encoder = LabelEncoder().fit(train_labels["label"].astype(str))
    x_train = feature_matrix(train_rows, train_ids, tubes, reducers, widths, args.bins, args.max_events)
    y_train = encoder.transform(train_labels["label"].astype(str))
    x_test = feature_matrix(test_rows, test_ids, tubes, reducers, widths, args.bins, args.max_events)
    y_test = test_labels["label"].astype(str).to_numpy()

    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=args.seed)
    rf.fit(x_train, y_train)
    pred_idx = rf.predict(x_test)
    proba = rf.predict_proba(x_test)
    pred = encoder.inverse_transform(pred_idx)
    metrics = save_predictions(out, test_ids, y_test, pred, proba, encoder.classes_)
    joblib.dump({"rf": rf, "encoder": encoder, "tubes": tubes, "widths": widths, "args": vars(args)}, out / "umap_rf_paper.joblib")
    write_json(out / "metadata.json", {"method": "paper_umap_rf", "args": vars(args), "tubes": tubes, "metrics": metrics})
    print(metrics, flush=True)


if __name__ == "__main__":
    main()

