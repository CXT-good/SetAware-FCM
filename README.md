# SetAware-FCM

Marker-aware and population-context models for flow cytometry, covering **cell-level classification** and **patient-level classification**.

## Models

| Component | Role | Implementation |
| --- | --- | --- |
| MAE | Masked marker-value pretraining | [mae](src/mae/) |
| CTransformer | Marker-aware cell representation and classification | [ctransformer](src/cell_level/ctransformer/) |
| PACTransformer | Population-context cell classification from cell embeddings | [pactransformer](src/cell_level/pactransformer/) |
| CA-MIL | Frozen MAE encoder with marker and cell gated-attention pooling | [ca_mil](src/patient_level/ca_mil/) |
| TACA-MIL | Within-tube Set Transformer followed by patient-level pooling | [taca_mil](src/patient_level/taca_mil/) |

Cell-level baselines include logistic regression, DeepCyTOF-style MLP, XGBoost, FlowSOM, and Set Transformer-only. Patient-level baselines include UMAP-RF, GMM-Fisher-SVM, Ensemble CNN, and FlowCat.

## Repository layout

```text
src/
├── dataio/
├── common/
├── mae/
├── cell_level/
│   ├── ctransformer/
│   ├── pactransformer/
│   ├── multicenter/
│   └── baselines/
│       ├── logistic_regression/
│       ├── deepcytof/
│       ├── xgboost/
│       ├── flowsom/
│       └── set_transformer/
└── patient_level/
    ├── ca_mil/
    ├── taca_mil/
    └── baselines/
        ├── umap_rf/
        ├── gmm_fisher_svm/
        ├── ensemble_cnn/
        └── flowcat/
```

Each model directory groups `model.py`, `train.py`, and `eval.py`. `common/` holds shared attention blocks and patient-baseline helpers; `dataio/` contains the marker vocabulary, class mappings, and data loaders. `cell_level/multicenter/` contains separate cross-centre workflows.

## Setup

The current repository publishes source code and this README, without package-installation metadata. Use Python 3.9 or newer, install the dependencies, and add `src/` to `PYTHONPATH`. The commands below use Bash.

```bash
git clone https://github.com/Intelligent-Medical-Research-Group/SetAware-FCM.git
cd SetAware-FCM
python -m venv .venv
source .venv/bin/activate
python -m pip install "torch>=2.0" "numpy>=1.23" "pandas>=1.5" \
  "scikit-learn>=1.5" "scipy>=1.10" "matplotlib>=3.7" \
  "openpyxl>=3.0" "tqdm>=4.64" "joblib>=1.2"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
```

For GPU experiments, choose a PyTorch build compatible with the CUDA environment. Optional baseline dependencies are:

```bash
python -m pip install "umap-learn>=0.5" "xgboost>=2.0"
```

FlowSOM uses the implementation included in this repository. There is no `pip install -e .` workflow in this source-only snapshot. Record the exact Python, PyTorch, CUDA, and dependency versions used for each experiment.

## Quick CPU check

This example needs no clinical data or pretrained weights:

```bash
python - <<'PY'
import torch
from cell_level.ctransformer.model import Stage1Transformer

torch.manual_seed(42)
model = Stage1Transformer(
    markers=["CD3", "CD4", "CD8"], embed_dim=16, num_heads=4,
    num_layers=1, dim_feedforward=32, num_classes=3, dropout=0.0,
).eval()
with torch.no_grad():
    logits, embeddings = model(torch.randn(4, 3), torch.arange(3))
print(tuple(logits.shape), tuple(embeddings.shape))
PY
```

Expected output: `(4, 3) (4, 16)`. These are random-weight outputs on synthetic values, not clinical predictions.

## Data preparation

The loaders consume prepared **CSV event tables**, not raw FCS files. Each row represents a cell; marker names must match the vocabulary in [marker_mapping.py](src/dataio/marker_mapping.py). Document the compensation, transformations, filtering, and missing-value handling used to prepare your data.

- **Cell tasks:** an XLSX split workbook with `train` and `test` sheets, each containing `csv_path`. Event CSVs also require a `label` column. All event labels must match the dataset's class mapping.
- **Class mappings:** the workbook basename determines the dataset key; for example, `example_dataset_split.xlsx` uses key `example`. Supply a JSON mapping with `datasets.example.class_list` and `class_counts`; counts are an ordered numeric list aligned with the class list. The bundled mapping contains aggregate research-dataset labels/counts and should be replaced for a different dataset.
- **Patient tasks:** each workbook row describes one tube, with `patient_id`, `csv_path`, and `label`; `tube_type` is optional. All rows for a patient must use the same diagnosis label. Keep every patient's cells and tubes in the same data partition.
- **Paths:** use valid local event paths in the workbooks. A separate `validation` sheet is needed for the patient-training examples below.

The mapping-generation entry point can scan your training split:

```bash
python -m cell_level.ctransformer.build_class_mappings --help
```

Pass `--sheet train` when generating mappings from a `train` sheet; the historical default is `total`.

## Training and evaluation

### Entry points

Append `--help` to inspect each command's required data paths and model settings.

| Component | Training | Evaluation |
| --- | --- | --- |
| MAE | `python -m mae.train` | `python -m mae.eval` after `mae.finetune` |
| CTransformer | `python -m cell_level.ctransformer.train` | `python -m cell_level.ctransformer.eval` |
| PACTransformer | `python -m cell_level.pactransformer.train` | `python -m cell_level.pactransformer.eval` |
| CA-MIL | `python -m patient_level.ca_mil.train` | `python -m patient_level.ca_mil.eval` |
| TACA-MIL | `python -m patient_level.taca_mil.train` | `python -m patient_level.taca_mil.eval` |

MAE pretraining requires CUDA/NCCL and at least two distributed processes on two GPUs. For example:

```bash
torchrun --standalone --nproc_per_node=2 -m mae.train \
  --pretrain_manifest /path/to/pretrain_manifest.xlsx \
  --model_size tiny --experiment_dir outputs/mae
```

`mae.eval` evaluates the supervised fine-tuned cell classifier, not masked reconstruction; its event labels must be integer-coded class indices. Weighted PACTransformer training and its formal evaluator select CUDA when available and otherwise use CPU.

### Cell-level pipeline

The data flow is **MAE pretraining → CTransformer training → cell-embedding export → PACTransformer training/evaluation**. CTransformer also supports explicitly random-initialized training with `--init_mode random`.

Export representations using a trained CTransformer:

```bash
python -m cell_level.ctransformer.extract_embeddings \
  --model_path /path/to/cell_checkpoint.pt \
  --data_dir /path/to/train_event_csvs \
  --output_dir outputs/train_payload
```

The exporter accepts a CSV file or a flat CSV directory. Export each split separately. Its matching `embeddings/`, `labels/`, and `xuhao/` directories contain cell embeddings `[N, D]`, labels `[N]`, and original row indices `[N]`. Preserve filenames and row alignment, and set PACTransformer's `--dim_input` to `D`.

### Patient-level example

With your own prepared patient workbook and a matching tiny MAE encoder checkpoint:

```bash
python -m patient_level.ca_mil.train \
  --split_xlsx /path/to/patient_split.xlsx \
  --train_sheet train --val_sheet validation \
  --pretrained_encoder /path/to/pretrained_encoder.pt \
  --model_size tiny --output_dir outputs/ca_mil

python -m patient_level.ca_mil.eval \
  --split_xlsx /path/to/patient_split.xlsx --sheet test \
  --checkpoint outputs/ca_mil/checkpoint_best.pt \
  --output_dir outputs/ca_mil/test
```

For TACA-MIL, use `patient_level.taca_mil.train` and `patient_level.taca_mil.eval` with the intended set-attention settings. Both patient models freeze the MAE encoder; provide the intended pretrained weights rather than freezing a randomly initialized encoder.

The patient trainers retain a historical `--val_sheet test` default. The example explicitly selects a separate `validation` sheet; it is a usage example, not a claim about historical paper runs. The default `--filter_mode count` retains patients with the modal tube count; use and record the filtering and cell-cap settings required by your experiment.

### Baselines

Use the method's own `train` and `eval` modules, for example:

```bash
python -m cell_level.baselines.logistic_regression.train --help
python -m cell_level.baselines.logistic_regression.eval --help
python -m patient_level.baselines.flowcat.train --help
python -m patient_level.baselines.flowcat.eval --help
```

Patient baseline evaluators restore artifacts produced by their matching trainers. GMM-Fisher-SVM supports pooled and per-tube variants; inspect its evaluator's `--variant` option. Cross-centre helpers are separate commands under `cell_level.multicenter`.

## Reproducibility and availability

Clinical event data, patient split manifests, trained checkpoints, and generated results are **not included**. Use data and models for which you have the appropriate access permissions. Exact result reproduction requires the original preprocessing, patient partitions, complete parameters, seeds, checkpoints, and checkpoint-selection policies; constructor defaults and the examples above do not identify final paper runs.

The source was checked during curation with 60 local tests, 43 CLI help checks, isolated package imports, and synthetic CPU runs. Those supporting tests are not included in this source-only repository, and the checks do not establish reproduction of clinical metrics. Historical randomness remains in some baseline workflows, including UMAP fitting and hash-derived sampling; record `PYTHONHASHSEED` and the saved artifacts when applicable.

Known limitation: the inherited standalone logistic-regression evaluator uses multiclass `argmax` decoding, which is incorrect for a binary scikit-learn checkpoint with one coefficient row. Do not use that path for binary LR results without correcting and validating its decoding separately.

Load `state_dict` checkpoints with the corresponding architecture and configuration. Historical whole-model Python pickles may embed obsolete module paths and are not guaranteed to load after code reorganization.

When referring to this implementation, record the [repository URL](https://github.com/Intelligent-Medical-Research-Group/SetAware-FCM) and the exact commit or release used. No software license is currently included; clarify reuse permissions and applicable third-party licenses with the repository owners.
