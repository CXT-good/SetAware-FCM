from pathlib import Path

import pandas as pd
import torch

from dataio import marker_mapping
from dataio.class_lists import load_class_mappings_json
from mae.model import MAEConfig, MAEEncoder
from patient_level.ca_mil import export_attention
from patient_level.ca_mil.eval import build_model as build_ca_model
from patient_level.taca_mil.eval import build_model
from patient_level.taca_mil.model import PatientLevelTubeSetTransformerMIL


def test_default_class_mapping_loads_outside_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FCM_ML_CLASS_MAPPING_PATH", raising=False)
    mappings = load_class_mappings_json(marker_mapping.default_class_mapping_path())
    assert mappings
    dataset_key, expected = next(iter(mappings.items()))
    classes, counts = marker_mapping.load_class_list_for_dataset(dataset_key)
    assert classes == expected.class_list
    assert counts == expected.class_counts


def test_taca_checkpoint_restores_without_original_pretrained_file(tmp_path):
    saved_args = {
        "pretrained_encoder": str(tmp_path / "missing_pretrained_encoder.pt"),
        "d_model": 8,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "n_heads": 2,
        "d_ff": 16,
        "dropout": 0.0,
        "set_hidden_dim": 8,
        "set_num_heads": 2,
        "set_num_inds": 2,
        "set_num_layers": 1,
        "attention_hidden_dim": 8,
        "classifier_hidden_dim": 8,
    }
    encoder = MAEEncoder(MAEConfig(
        num_markers=len(marker_mapping.MARKER_LIST),
        seq_len=len(marker_mapping.MARKER_LIST),
        d_model=8, encoder_layers=1, decoder_layers=1, n_heads=2,
        d_ff=16, dropout=0.0,
    ))
    original = PatientLevelTubeSetTransformerMIL(
        encoder, num_classes=2, set_hidden_dim=8, set_num_heads=2,
        set_num_inds=2, set_num_layers=1, attention_hidden_dim=8,
        classifier_hidden_dim=8, dropout=0.0,
    ).eval()
    payload = {
        "args": saved_args,
        "label_to_idx": {"synthetic_a": 0, "synthetic_b": 1},
        "model_state_dict": original.state_dict(),
    }
    restored, labels, restored_args = build_model(payload, torch.device("cpu"))
    tubes = [{
        "marker_ids": torch.tensor([0, 1]),
        "values": torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
        "csv_path": "synthetic.csv",
        "tube_type": "synthetic",
        "marker_names": marker_mapping.MARKER_LIST[:2],
        "row_ids": torch.tensor([0, 1]),
    }]
    with torch.no_grad():
        torch.testing.assert_close(restored(tubes)["logits"], original(tubes)["logits"])
    assert labels == payload["label_to_idx"]
    assert restored_args["pretrained_encoder"] == saved_args["pretrained_encoder"]
    assert not Path(saved_args["pretrained_encoder"]).exists()


def test_ca_attention_export_restores_relocated_full_checkpoint(tmp_path, monkeypatch):
    saved_args = {
        "pretrained_encoder": str(tmp_path / "missing_pretrained_encoder.pt"),
        "d_model": 8, "encoder_layers": 1, "decoder_layers": 1,
        "n_heads": 2, "d_ff": 16, "dropout": 0.0,
        "attention_hidden_dim": 8, "classifier_hidden_dim": 8,
    }
    payload = {"args": saved_args, "label_to_idx": {"synthetic": 0}}
    model, _, _ = build_ca_model(payload, torch.device("cpu"))
    payload["model_state_dict"] = model.state_dict()
    checkpoint = tmp_path / "patient.pt"
    torch.save(payload, checkpoint)
    csv_path = tmp_path / "synthetic_tube1.csv"
    pd.DataFrame({marker: [0.5, 1.0] for marker in marker_mapping.MARKER_LIST[:2]}).to_csv(csv_path, index=False)
    workbook = tmp_path / "synthetic_split.xlsx"
    pd.DataFrame({
        "patient_id": ["synthetic_patient"], "csv_path": [str(csv_path)],
        "label": ["synthetic"],
    }).to_excel(workbook, sheet_name="test", index=False)
    output = tmp_path / "attention.csv"
    monkeypatch.setattr("sys.argv", [
        "export_attention", "--checkpoint", str(checkpoint),
        "--split_xlsx", str(workbook), "--output_csv", str(output),
        "--filter_mode", "none", "--device", "cpu",
    ])
    export_attention.main()
    exported = pd.read_csv(output)
    assert len(exported) == 2
    assert exported["patient_id"].tolist() == ["synthetic_patient"] * 2
    assert abs(exported["cell_attention_weight"].sum() - 1.0) < 1e-6
    assert not Path(saved_args["pretrained_encoder"]).exists()
