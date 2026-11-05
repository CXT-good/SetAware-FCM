"""Exercise the five proposed architectures on synthetic CPU tensors."""

from __future__ import annotations

import json

import torch

from mae.model import MAEConfig, MAEEncoder, MAEModel
from cell_level.ctransformer.model import Stage1Transformer
from cell_level.pactransformer.model import SetTransformer
from patient_level.ca_mil.model import PatientLevelMIL
from patient_level.taca_mil.model import PatientLevelTubeSetTransformerMIL


def run_demo() -> dict:
    torch.manual_seed(42)
    torch.set_num_threads(1)
    cells, markers, dimension, classes = 8, 6, 16, 3
    marker_ids = torch.arange(markers)
    values = torch.randn(cells, markers)
    config = MAEConfig(
        num_markers=markers, seq_len=markers, d_model=dimension,
        encoder_layers=1, decoder_layers=1, n_heads=4, d_ff=32, dropout=0,
    )
    mae = MAEModel(config).eval()
    mask = torch.zeros(cells, markers, dtype=torch.bool)
    mask[:, -2:] = True
    cell_model = Stage1Transformer(
        markers=list(range(markers)), embed_dim=dimension, num_heads=4,
        num_layers=1, dim_feedforward=32, dropout=0, num_classes=classes,
    ).eval()
    population_model = SetTransformer(
        dim_input=dimension, dim_hidden=dimension, num_heads=4,
        num_inds=4, num_layers=1, num_classes=classes, dropout=0,
    ).eval()
    tubes = []
    for tube_index in range(2):
        tubes.append({
            "marker_ids": marker_ids,
            "values": values + tube_index * 0.1,
            "csv_path": f"synthetic_tube_{tube_index}.csv",
            "tube_type": f"synthetic_panel_{tube_index}",
            "marker_names": [f"marker_{index}" for index in range(markers)],
            "row_ids": torch.arange(cells),
        })
    patient_model = PatientLevelMIL(
        MAEEncoder(config), num_classes=classes, attention_hidden_dim=16,
        classifier_hidden_dim=16, dropout=0,
    ).eval()
    context_patient_model = PatientLevelTubeSetTransformerMIL(
        MAEEncoder(config), num_classes=classes, set_hidden_dim=dimension,
        set_num_heads=4, set_num_inds=4, set_num_layers=1,
        attention_hidden_dim=16, classifier_hidden_dim=16, dropout=0,
    ).eval()
    with torch.no_grad():
        reconstruction, returned_mask = mae(marker_ids.expand(cells, -1), values, mask=mask)
        cell_logits, embeddings = cell_model(values, marker_ids)
        population_logits = population_model(embeddings.unsqueeze(0))
        patient = patient_model(tubes, return_attention=True)
        context_patient = context_patient_model(tubes, return_attention=True)
    expected = [
        (reconstruction, (cells, markers)),
        (cell_logits, (cells, classes)),
        (population_logits, (1, cells, classes)),
        (patient["logits"], (classes,)),
        (context_patient["logits"], (classes,)),
    ]
    for tensor, shape in expected:
        assert tuple(tensor.shape) == shape
        assert torch.isfinite(tensor).all()
    assert torch.equal(mask, returned_mask)
    for prediction in [patient, context_patient]:
        assert torch.allclose(prediction["cell_attention"].sum(), torch.tensor(1.0), atol=1e-6)
    assert all(not parameter.requires_grad for parameter in patient_model.encoder.parameters())
    return {
        "data": "synthetic; random model weights; not clinical predictions",
        "MAE_reconstruction": list(reconstruction.shape),
        "CTransformer_logits": list(cell_logits.shape),
        "PACTransformer_logits": list(population_logits.shape),
        "CA_MIL_logits": list(patient["logits"].shape),
        "TACA_MIL_logits": list(context_patient["logits"].shape),
        "status": "passed",
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
