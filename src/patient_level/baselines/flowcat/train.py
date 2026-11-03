from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from common.patient_paper import (
    patient_labels,
    patient_table,
    save_predictions,
    write_json,
)


from patient_level.baselines.flowcat.model import (
    FlowCatCNN,
    TensorDataset,
    patient_tensor,
    predict,
    tube_order,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-style FlowCat: SOM maps per panel followed by CNN classifier.")
    parser.add_argument("--split_xlsx", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_events", type=int, default=12000)
    parser.add_argument("--max_som_events", type=int, default=2500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_rows = patient_table(args.split_xlsx, "train")
    test_rows = patient_table(args.split_xlsx, "test")
    train_labels = patient_labels(train_rows)
    test_labels = patient_labels(test_rows)
    tubes = tube_order(train_rows)
    encoder = LabelEncoder().fit(train_labels["label"].astype(str))
    train_ids = train_labels["patient_id"].astype(str).tolist()
    test_ids = test_labels["patient_id"].astype(str).tolist()
    y_train = encoder.transform(train_labels["label"].astype(str))
    y_test = test_labels["label"].astype(str).to_numpy()
    print({"device": args.device, "tubes": tubes, "classes": encoder.classes_.tolist()}, flush=True)

    x_train = np.stack([patient_tensor(train_rows, pid, tubes, args.grid_size, args.max_events, args.max_som_events, args.seed) for pid in train_ids])
    x_test = np.stack([patient_tensor(test_rows, pid, tubes, args.grid_size, args.max_events, args.max_som_events, args.seed) for pid in test_ids])

    model = FlowCatCNN(in_channels=len(tubes), num_classes=len(encoder.classes_)).to(args.device)
    counts = np.bincount(y_train, minlength=len(encoder.classes_)).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32, device=args.device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print({"epoch": epoch, "loss": float(np.mean(losses))}, flush=True)

    proba = predict(model, x_test, args.device, args.batch_size)
    pred = encoder.inverse_transform(np.argmax(proba, axis=1))
    metrics = save_predictions(out, test_ids, y_test, pred, proba, encoder.classes_)
    torch.save({"state_dict": model.state_dict(), "tubes": tubes, "classes": encoder.classes_.tolist(), "args": vars(args)}, out / "flowcat_paper.pt")
    joblib.dump({"encoder": encoder, "tubes": tubes, "args": vars(args)}, out / "flowcat_paper.joblib")
    write_json(out / "metadata.json", {"method": "paper_flowcat", "args": vars(args), "tubes": tubes, "metrics": metrics})
    print(metrics, flush=True)


if __name__ == "__main__":
    main()

