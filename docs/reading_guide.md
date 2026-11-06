# Reading guide

## Model and data flow

Start with `dataio.marker_mapping` and `dataio.class_lists`. Marker order and label order must agree with the selected data and checkpoints.

MAE projects values, adds marker embeddings, masks values, and reconstructs them with a Transformer encoder/decoder. The training mask ratio is explicit; a constructor default does not identify a paper recipe. `mae.finetune` and `mae.eval` provide supervised cell classification using the pretrained encoder; `mae.eval` is not a masked-reconstruction evaluator.

CTransformer adds a CLS token to marker-value tokens and predicts one label per cell. Its exported representations feed PACTransformer, which models the cells within a tube with induced set attention and returns one output per cell.

CA-MIL reads event CSVs through `dataio.patient_dataset`. A frozen MAE encoder supplies marker features; marker attention forms cell features and cell attention pools them into a patient representation. TACA-MIL additionally updates cells with Set Transformer blocks within each tube before patient pooling. A complete patient checkpoint contains the frozen encoder state.

## Where to read

| Path under `src/` | Contents |
| --- | --- |
| `dataio/` | Marker/class metadata, CSV loaders, patient data, payload loading |
| `common/attention.py` | Gated attention shared by the patient models |
| `common/set_attention.py` | Shared set-attention blocks |
| `common/patient_baselines.py`, `common/patient_paper.py` | Patient baseline data, features, and training helpers |
| `mae/` | MAE model, pretraining, supervised fine-tuning/evaluation |
| `cell_level/ctransformer/` | CTransformer model, training/evaluation, embedding export |
| `cell_level/pactransformer/` | PACTransformer model, weighted training/formal evaluation, retained variants |
| `patient_level/ca_mil/` | CA-MIL model, training, evaluation |
| `patient_level/taca_mil/` | TACA-MIL model, training, evaluation |

Each model directory groups `model.py`, `train.py`, and `eval.py`. Run the installed top-level packages with `python -m`, as shown in [training](training.md).

## Baselines

Cell methods live in `cell_level/baselines/{logistic_regression,deepcytof,xgboost,flowsom,set_transformer}/`. Cross-centre helpers live in `cell_level/multicenter/`. Patient methods live in `patient_level/baselines/{umap_rf,gmm_fisher_svm,ensemble_cnn,flowcat}/`.

Use the method's `train` and `eval` modules, for example:

```bash
python -m cell_level.baselines.logistic_regression.train --help
python -m cell_level.baselines.logistic_regression.eval --help
python -m patient_level.baselines.flowcat.train --help
python -m patient_level.baselines.flowcat.eval --help
```

Patient standalone evaluators restore the artifacts written by their matching trainers. They do not retrain the method before evaluation.

## Weights and provenance

Use the matching architecture/configuration when loading a `state_dict` checkpoint. Historical whole-model Python pickles may contain the former module paths and are not guaranteed to load after reorganization. Complete CA-MIL/TACA-MIL checkpoints can restore their saved encoder without the original pretraining file.

This release adapts imports, package layout, portable paths, checkpoint restoration, and command interfaces. Source provenance is recorded in [source_map.csv](source_map.csv), and actual software checks are recorded in [validation](validation.md). Clinical training and paper metrics require the original data, preprocessing, split, seed, and checkpoint-selection records.
