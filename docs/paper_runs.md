# Paper experiment configurations

This directory separates executable manuscript schedules from recovered historical run records. It does not substitute a new random split or a newly trained checkpoint for a reported paper result. No clinical event rows, patient identifiers, original split workbooks, prediction rows, or model weights are included.

## What is provided

| Files | Meaning |
| --- | --- |
| `configs/paper/manuscript_*.json` | Five executable architecture/schedule recipes derived from Methods 2.3 of the manuscript inspected on 2026-11-05. They are not identified final runs. |
| `configs/paper/historical/*_train.json` | Eight patient-level training configurations recovered from the evaluation metadata selected by the paper source workbook. |
| `configs/paper/historical/*_eval.json` | Eight matching evaluation recipes with explicit effective cohort, cell-cap and seed controls. |
| `configs/paper/evidence.json` | Sanitized provenance, metadata SHA-256 values, checkpoint roles/epochs, recovered arguments and unresolved inputs. |
| `configs/patient_{ca_mil,taca_mil}_evaluate.example.json` | Portable matched-control evaluation examples for the separate-validation examples, not historical paper runs. |

Every paper recipe sets `clinical_reproduction_verified` to `false`. Configuration recovery and CLI validation are not rerunning the clinical experiment. A file hash identifies an inspected record. It does not make a private record public or prove that an unavailable checkpoint has been reproduced.

## Source evidence and paper mapping

The inspected manuscript has Methods 2.3 for training and 2.4 for metrics, Table 2 for cell classification, Table 3 for patient classification, and Table 4 for cross-centre transfer. `evidence.json` records the inspected manuscript hash and the hash of `patient_and_gate_level_large_tables.xlsx`, sheet `Sources`, together with exact source-row numbers. Both original documents are maintained outside this software release.

The workbook identifies all eight proposed-model patient table sources. Their adjacent `eval_config.json` files retain saved training arguments, and their `summary.json` files record effective evaluation settings and checkpoint epochs. Only whitelisted nonidentifying configuration fields and checksums are exported. One selected checkpoint was available for a weights-only metadata check, the ZZ6B TACA-MIL epoch-156 checkpoint. No checksum is invented for the seven unavailable selected patient checkpoints or the unavailable selected cell checkpoints.

| Dataset | CA-MIL checkpoint | TACA-MIL checkpoint | TACA-MIL layers / inducing points / seed / weight decay |
| --- | --- | --- | --- |
| Z2JW (`09_Z2JW_LMD`) | latest, epoch 160 | best, epoch 54 | 1 / 8 / 2024 / 0.0001 |
| ZZ6B (`14_ZZ6B`) | latest, epoch 160 | best, epoch 156 | 1 / 8 / 7 / 0.001 |
| ZZYA (`15_ZZYA`) | latest, epoch 160 | latest, epoch 160 | 2 / 16 / 42 / 0.0001 |
| Z27 | latest, epoch 160 | best, epoch 36 | 2 / 16 / 42 / 0.0001 |

All recovered patient runs use tiny MAE encoders, 160 training epochs, learning rate 0.0001, a training cap of 5,000 cells per tube, and `filter_mode=count`. CA-MIL uses seed 42 and weight decay 0.0001 in all four datasets. TACA-MIL uses hidden dimension 128 and four heads. These dataset variants take precedence over a generic manuscript-default recipe when tracing a particular table source.

### Historical checkpoint selection

All eight recovered training records use `train_sheet=train` and `val_sheet=test`. The trainer monitors macro-F1 on that selection sheet for its best checkpoint. Historical recipes retain this fact rather than renaming the test sheet to validation. A best checkpoint selected this way is not an untouched independent-test estimate. The record alone also does not establish whether a separately reported latest checkpoint was chosen without inspecting test results.

For a new evaluation, create patient-disjoint training, validation and test partitions and fix checkpoint selection before test evaluation. The `manuscript_*_mil_train.json` and top-level `*.example.json` training recipes use a distinct `validation` sheet for this prospective workflow. That is a different protocol from the historical run and must not be described as reconstructing the same original split.

### Nullable CLI options are not the effective historical cohort

The Z2JW and Z27 TACA-MIL historical evaluation CLI records contain null filtering, cell-cap and seed options. Their associated summaries nevertheless record `filter_mode=count`, a 5,000-cell cap and the saved training seed. The old evaluator restored these values. The released evaluator has different omission semantics, so the historical evaluation recipes pass the recorded effective values explicitly. For ZZ6B, a recorded null seed becomes zero in the dataset, and evaluation truncation is nonrandom. All recovered evaluation summaries have `random_sample=false`, meaning the first 5,000 rows are used rather than a new random draw.

The recovered CA-MIL and TACA-MIL evaluation recipes therefore use the same explicit `count` cohort and 5,000-cell cap within each dataset. This does not by itself prove identical patients without the original private split workbook. `count` retains the modal tube count and can exclude other patients, so cohort counts must be checked rather than assumed.

## Running a recipe

Run from the repository root after installing the documented environment. Set `FCM_DATA_ROOT` and `FCM_OUTPUT_ROOT` to your own authorized input and result directories. Dry-run mode prints the command without opening data or checkpoints and leaves missing environment variables visible:

```bash
python scripts/run_experiment.py configs/paper/manuscript_mae_pretrain.json --dry-run
python scripts/run_experiment.py configs/paper/historical/14_ZZ6B_taca_mil_train.json --dry-run
```

Execution refuses missing environment substitutions. A historical patient training recipe additionally requires `FCM_MAE_CHECKPOINT`, the intended pretrained encoder. Evaluation requires `FCM_PATIENT_CHECKPOINT`, the selected model file corresponding to that dataset, method and checkpoint role. An arbitrary available file is not a replacement for the selected file. Inspect the recipe's `unresolved` list before running:

```bash
export FCM_MAE_CHECKPOINT=/authorized/artifacts/pretrained_encoder.pt
export FCM_PATIENT_CHECKPOINT=/authorized/artifacts/selected_patient_checkpoint.pt
python scripts/run_experiment.py configs/paper/historical/14_ZZ6B_taca_mil_eval.json \
  --record-dir "$FCM_OUTPUT_ROOT/zz6b_taca_mil_provenance"
```

The manuscript-only CTransformer recipe requires `FCM_CELL_INIT_MODE=random` or `from_pretrain` and an explicitly supplied `FCM_TRAIN_SEED`. For random initialization, `FCM_MAE_CHECKPOINT=unused` satisfies substitution and the trainer ignores that path. For pretrained initialization, it must identify the actual compatible encoder. The source workbook's cell-model directory names mention `random` or `random_2`; these are path hints, not inspected checkpoint metadata, and do not verify the final initialization or seed. Do not assume that every paper cell result used an MAE-initialized classifier.

The manuscript PACTransformer recipe describes the Zhongnan 512-hidden, 8-head, 6-layer variant only. It requires `FCM_NUM_CLASSES` and `FCM_TRAIN_SEED`. Do not apply it unchanged to other datasets. Export train and held-out CTransformer embeddings separately, using the same chosen classifier checkpoint, marker mapping and class mapping, as described in [training](training.md).

## What remains author-supplied

- Original private partition identities, preprocessing/marker-mapping identities and pretrained encoder identities. The manuscript's 80/20 patient-disjoint allocation does not disclose the original split seed or membership.
- Seven selected patient checkpoint files/checksums and the selected cell-model checkpoint metadata. Recovered patient arguments and roles are available even when the selected file is not.
- Exact historical environment build, per-run process count and complete initialization/sampling state. A pinned new release environment is not evidence of an identical historical binary environment.
- Complete dataset-specific cell and baseline hyperparameters, cached-embedding identities, and baseline fitted-artifact checksums. The source workbook locates model outputs but is not a complete hyperparameter log.
- A historical source/configuration-to-checkpoint-to-table manifest for all baselines and Table 4. Cross-centre comparisons involved target-centre checkpoint selection and remain exploratory transfer estimates, not prospective independent-site validation.
- A clinical rerun and numerical comparison against the paper's metrics. The release regression tests use synthetic or nonclinical inputs only.

These gaps are machine-readable in the evidence and recipe files. A paper-specific software tag fixes the released code version, not missing data or unverified historical results. Restricted clinical measurements can remain private; their absence should be distinguished from recoverable configuration metadata.
