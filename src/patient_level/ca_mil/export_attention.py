from __future__ import annotations

import argparse
import csv
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.marker_mapping import MARKER_LIST
from dataio.patient_dataset import PatientLevelDataset, patient_collate
from patient_level.ca_mil.model import PatientLevelMIL, build_encoder_from_checkpoint


def move_patient_to_device(sample: dict, device: torch.device) -> dict:
    for tube in sample["tubes"]:
        tube["values"] = tube["values"].to(device, non_blocking=True)
        tube["marker_ids"] = tube["marker_ids"].to(device, non_blocking=True)
        tube["row_ids"] = tube["row_ids"].to(device, non_blocking=True)
    return sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export patient-level MIL cell and marker attention.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split_xlsx", required=True)
    p.add_argument("--sheet", default="test")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--max_cells_per_tube", type=int, default=0)
    p.add_argument("--filter_mode", choices=["none", "count", "signature"], default="count")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    encoder = build_encoder_from_checkpoint(
        "",
        num_markers=len(MARKER_LIST),
        seq_len=len(MARKER_LIST),
        d_model=train_args["d_model"],
        encoder_layers=train_args["encoder_layers"],
        decoder_layers=train_args["decoder_layers"],
        n_heads=train_args["n_heads"],
        d_ff=train_args["d_ff"],
        dropout=train_args["dropout"],
        device=device,
    )
    model = PatientLevelMIL(
        encoder,
        num_classes=len(label_to_idx),
        attention_hidden_dim=train_args["attention_hidden_dim"],
        classifier_hidden_dim=train_args["classifier_hidden_dim"],
        dropout=train_args["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    ds = PatientLevelDataset(
        args.split_xlsx,
        args.sheet,
        label_to_idx=label_to_idx,
        filter_mode=args.filter_mode,
        max_cells_per_tube=args.max_cells_per_tube,
        random_sample=False,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=patient_collate, num_workers=0)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    fields = [
        "patient_id",
        "tube_name",
        "csv_path",
        "cell_id",
        "cell_attention_weight",
        "marker_attention_weights",
        "original_marker_values",
        "predicted_class",
        "true_label",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        with torch.no_grad():
            for sample in tqdm(loader, desc="export"):
                sample = move_patient_to_device(sample, device)
                out = model(sample["tubes"], return_attention=True)
                probs = torch.softmax(out["logits"], dim=0)
                pred_idx = int(probs.argmax().item())
                pred_label = idx_to_label[pred_idx]
                cell_attention = out["cell_attention"].detach().cpu()

                offset = 0
                rows = []
                for meta, marker_attn in zip(out["metadata"], out["marker_attention_by_tube"]):
                    n = meta["num_cells"]
                    tube_cell_attn = cell_attention[offset : offset + n]
                    offset += n
                    marker_attn_cpu = marker_attn.detach().cpu()
                    values_cpu = meta["values"].detach().cpu()
                    row_ids = meta["row_ids"].detach().cpu()
                    marker_names = meta["marker_names"]
                    for i in range(n):
                        rows.append(
                            {
                                "patient_id": sample["patient_id"],
                                "tube_name": meta["tube_type"],
                                "csv_path": meta["csv_path"],
                                "cell_id": int(row_ids[i].item()),
                                "cell_attention_weight": float(tube_cell_attn[i].item()),
                                "marker_attention_weights": json.dumps(
                                    dict(zip(marker_names, [float(x) for x in marker_attn_cpu[i].tolist()])),
                                    ensure_ascii=False,
                                ),
                                "original_marker_values": json.dumps(
                                    dict(zip(marker_names, [float(x) for x in values_cpu[i].tolist()])),
                                    ensure_ascii=False,
                                ),
                                "predicted_class": pred_label,
                                "true_label": sample["label_name"],
                            }
                        )
                rows.sort(key=lambda r: r["cell_attention_weight"], reverse=True)
                for row in rows[: args.top_k if args.top_k > 0 else None]:
                    writer.writerow(row)


if __name__ == "__main__":
    main()

