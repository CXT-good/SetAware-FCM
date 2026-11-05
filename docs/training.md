# Training and evaluation

Install with `python -m pip install -e .` before running these modules. Paths and recipes below are examples; set the required environment variables and edit dataset-specific parameters before executing them.

## MAE

```bash
torchrun --standalone --nproc_per_node=2 \
  -m mae.train \
  --pretrain_manifest "$FCM_DATA_ROOT/pretrain_manifest.xlsx" \
  --model_size tiny --mask_ratio 0.25 --epochs 80 \
  --experiment_dir "$FCM_OUTPUT_ROOT/mae"
```

Pretraining requires at least two CUDA GPUs and NCCL. The manuscript describes eight-GPU pretraining, as recorded in `configs/mae_pretrain.example.json`. Match process count and per-device batch size to the intended run; changing them changes effective batch size.

Supervised MAE classification is a separate workflow:

```bash
python -m mae.finetune --help
python -m mae.eval --help
```

`mae.eval` evaluates the supervised classifier checkpoint, not masked reconstruction.

## CTransformer

```bash
python -m cell_level.ctransformer.train \
  --train_xlsx "$FCM_DATA_ROOT/cell_split.xlsx" \
  --class_mapping_path "$FCM_DATA_ROOT/class_mapping.json" \
  --model_size tiny --init_mode from_pretrain \
  --pretrained_path "$FCM_MAE_CHECKPOINT" \
  --output_dir "$FCM_OUTPUT_ROOT/cell_stage1"
python -m cell_level.ctransformer.eval --help
```

`--train_xlsx` is the split workbook argument. `--init_mode random` enables an explicitly random-initialized comparison. Early stopping and best-checkpoint selection monitor average training loss; record that selection rule in the experiment manifest.

Export cell representations before PACTransformer training:

```bash
python -m cell_level.ctransformer.extract_embeddings \
  --model_path "$FCM_CELL_CHECKPOINT" \
  --data_dir "$FCM_DATA_ROOT/events/train" \
  --output_dir "$FCM_OUTPUT_ROOT/train_payload"
```

The exporter reads CSVs, not the split workbook. Supply the intended partition's CSV directory and repeat separately for held-out partitions. Outputs are `embeddings/`, `labels/`, and `xuhao/`; see [data formats](data_format.md).

## PACTransformer

```bash
python -m cell_level.pactransformer.train --help
python -m cell_level.pactransformer.eval --help
python scripts/run_experiment.py configs/cell_stage2.example.json --dry-run
python scripts/run_experiment.py configs/cell_stage2.example.json \
  --record-dir "$FCM_OUTPUT_ROOT/cell_stage2/provenance"
python scripts/run_experiment.py configs/cell_evaluate.example.json
```

Set `FCM_NUM_CLASSES`, `FCM_DATA_ROOT`, `FCM_OUTPUT_ROOT`, and `FCM_CELL_CHECKPOINT` as needed. Canonical `train` uses class-weighted training and writes best/latest checkpoints, selecting best by training loss. `train_basic`, `eval_basic`, and `train_safe_sample` retain separate historical workflows; sampling changes must be recorded.

The example uses a Zhongnan-style architecture and must be checked for the intended dataset. The weighted trainer and canonical formal `eval` use CUDA when available and otherwise support CPU. Full-sized tube inference can still require substantial memory; the synthetic demo checks the architectures with small CPU inputs.

## Patient models

```bash
python -m patient_level.ca_mil.train --help
python -m patient_level.ca_mil.eval --help
python -m patient_level.taca_mil.train --help
python -m patient_level.taca_mil.eval --help
python scripts/run_experiment.py configs/patient_ca_mil.example.json --dry-run
python scripts/run_experiment.py configs/patient_taca_mil.example.json --dry-run
python scripts/run_experiment.py configs/patient_evaluate.example.json --dry-run
```

Both models freeze the MAE encoder during training. Supply the intended `--pretrained_encoder`: omitting it freezes randomly initialized encoder weights in the historical implementation. The examples require `FCM_MAE_CHECKPOINT` explicitly.

The trainers historically default `--val_sheet` to `test` and select their best checkpoint using that sheet's macro-F1. Example recipes explicitly use a separate `validation` sheet. This example protocol does not establish which splits were used in historical paper runs. Use the actual split and checkpoint-selection record to reproduce a reported result.

TACA-MIL set depth, weight decay, checkpoint epoch, cell cap, and tube filtering are experiment-specific. Training and evaluation defaults differ; the evaluation recipe makes cohort/cell-cap choices explicit. A complete patient checkpoint includes its encoder state and can restore without locating the original pretraining file.

## Baselines and cross-centre workflows

Each baseline model has a `train` and `eval` module. Append `--help` to inspect its own arguments.

| Method | Module prefix |
| --- | --- |
| Logistic regression | `cell_level.baselines.logistic_regression` |
| DeepCyTOF-style MLP | `cell_level.baselines.deepcytof` |
| XGBoost | `cell_level.baselines.xgboost` |
| FlowSOM | `cell_level.baselines.flowsom` |
| Set Transformer only | `cell_level.baselines.set_transformer` |
| Patient UMAP-RF | `patient_level.baselines.umap_rf` |
| Patient GMM-Fisher-SVM | `patient_level.baselines.gmm_fisher_svm` |
| Patient Ensemble CNN | `patient_level.baselines.ensemble_cnn` |
| Patient FlowCat | `patient_level.baselines.flowcat` |

For example, run `python -m patient_level.baselines.flowcat.eval --help`. Patient baseline evaluators load the matching trainer's saved artifacts using `--split_xlsx`, `--model_dir`, and `--output_dir`; `--split` defaults to `test`. Some trainers also evaluate their test split during training.

Cross-centre helpers are separate workflows under `cell_level.multicenter`: inspect `traditional`, `neural_and_flowsom`, or `reuse_stage1` with `python -m MODULE --help`.

FlowCat/GMM sampling uses stable logical patient/tube identities rather than Python hash randomization, and per-file GMM sampling is independent of input location. These repairs can change historical features; keep the original source revision and saved artifacts for comparisons. Some UMAP fitting paths remain unseeded. Record runtime seeds, saved artifacts, sampling settings and the chosen source revision with each run.

Cross-centre paper estimates used retrospective selection involving target-centre results. Preserve that provenance as exploratory transfer performance. A source-only checkpoint-selection protocol is a different experiment.

## Experiment and checkpoint records

Use `configs/experiment_manifest.template.csv` to link each reported result to its actual data partition, preprocessing, complete command, seed, checkpoint path/epoch and role, selection rule, and evaluation settings. See [paper run configurations](paper_runs.md) for recovered historical settings and manuscript-derived recipes. The separate `*.example.json` files are portable examples, not identities of the final paper checkpoints.

The manuscript describes patient-disjoint 80/20 train/test partitions. Reuse the actual partitions; a new seed or validation split defines a new protocol. Record patient eligibility, tube filtering, cell caps, and sampling. Keep the target metric's averaging convention: mean one-versus-rest balanced accuracy and scikit-learn's multiclass balanced accuracy are different quantities.

Load `state_dict` checkpoints with the corresponding model configuration. Historical whole-model Python pickles can embed obsolete import paths and are not guaranteed to load after the layout change. See [validation](validation.md) for completed software checks; those checks do not establish clinical performance reproduction.
