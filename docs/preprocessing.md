# Preprocessing and split protocol

## What this release starts from

The model pipelines start from **prepared, numeric per-tube event CSVs**, not raw FCS files. No clinical event tables, patient identifiers, historical split workbooks, or trained clinical checkpoints are distributed. The artificial generator is an interface example only; its values, labels and partition sizes do not represent any study cohort.

The manuscript states a patient-level, disease-stratified split with target proportions of 80% training and 20% testing for both tasks, keeping every patient's cells and tubes together. This describes the reported study design; it is not enough to recover exact historical membership. The available source does not establish a single historical split seed or a complete universal FCS-to-CSV procedure. Do not substitute a fresh random split and call it the original split. Small-cohort integer counts and any dataset-specific exceptions must be checked against the original, access-controlled manifest.

The six reported cell-level cohorts are ZZPH, BLAST110, Zhongnan, Bln, Bue and Routine; the patient-level cohorts are Z2JW, ZZ6B, ZZYA and Z27. Their marker panels and label spaces differ. The bundled cell class mapping records aggregate class definitions/counts; it is not a cohort-membership file or authorization to redistribute underlying data.

## External preparation record

For each dataset, retain a local preparation record containing the following **actual** choices. If an item cannot be recovered, mark it unavailable rather than supplying a plausible value.

| Stage | Required record |
| --- | --- |
| Acquisition/export | Dataset version, permitted access, instrument/export software and version, FCS channel names, event ordering and stable local event-row identity. |
| Compensation | Whether compensation was already applied; source and version of spillover/compensation information; software and settings. Avoid applying compensation twice. |
| Channel naming | Exact source-channel-to-released-marker aliases; scatter/viability channels retained or removed; duplicate-channel handling. Do not merge biologically different measurements merely because names resemble one another. |
| Transformation | The actual per-dataset transform, parameters/cofactors, clipping bounds, and whether values were transformed before export. If there was no transform, record that explicitly. |
| Event selection | Actual debris/doublet/viability or other exclusions and gate definitions; annotate expert cell-population labels separately from patient diagnosis. Preserve the mapping to original rows. |
| Numeric policy | Missing, malformed and infinite-value handling, dtype, and any scaling parameters and their fitting population. |
| Cohort construction | Deidentified patient linkage across all tubes/visits, inclusion/exclusion rules, disease stratification categories, and before/after counts. |
| Split provenance | Frozen train/validation/test membership, generation code/version and seed if known, duplicate reconciliation, any small-cohort exception, and a local checksum of each final manifest. |

No universal arcsinh cofactor, percentile normalization or gate threshold is specified by the core model loaders. A plot or explanation script applying a transformation for visualization is not evidence that all model inputs used that transformation. Do not re-transform an already transformed CSV or fit a cross-patient preprocessing model on held-out patients.

## What the public loaders actually do

- `dataio.marker_mapping` defines the case-sensitive global marker vocabulary. MAE, CTransformer and CA-MIL/TACA-MIL use recognized columns present in the CSV, allowing different tubes to have different panels; absent markers are not automatically measured or biologically imputed.
- `dataio.mae_dataset` converts selected marker values to float32, coercing malformed entries to NaN and replacing NaN with zero. `dataio.flow_dataset` similarly converts marker values, excludes `Time`/`event_ID`/the event label from automatic features, and falls back to class 0 for unknown or empty cell labels. These fallbacks can hide preparation errors; use the stricter preflight validator before model commands.
- `dataio.patient_dataset` groups tube rows by patient and enforces one patient label. `filter_mode=count` retains the modal tube count; `signature` retains the modal ordered tube signature; `none` does not apply that cohort filter. Record resulting eligible patient IDs/counts. These options alter the cohort, not merely performance.
- The patient dataset caps cells per tube when requested and uses a seed plus patient/tube indices for subsampling. Retain manifest ordering, the cap, random-versus-leading-row policy, seed and sampled original row IDs. The manuscript describes at most 5,000 cells per tube; an evaluation using a different cap is a different input protocol.
- Cell baselines have model-specific feature alignment: `dataio.baseline_loaders` uses marker intersections for LR/DeepCyTOF and a union with zero filling for XGBoost. Its current marker-set scan considers train and test headers. Record that behavior for historical runs; do not confuse header alignment with learning intensity-normalization parameters on held-out patients.
- Patient baselines perform their own feature selection and representation construction. For example, `common.patient_paper.scaled_values` uses per-tube 1st/99th percentile scaling for histogram/SOM features, while GMM-Fisher-SVM fits clipping/scaling from sampled training cells. These are method-specific operations, not a universal preprocessing recipe for the Transformer models. Some numeric-column baseline readers do not exclude event labels/metadata automatically; use feature-only CSVs and inspect the recorded feature list.

These statements describe source behavior, not a claim that the missing original FCS preparation has been recovered. Exact study reproduction requires the authorized original preparation and split records in addition to these model implementations.

## Freeze and validate a split

1. Link all tubes/visits belonging to the same person before splitting, using local identifiers that are not published. Assign a stable pseudonym for local computational manifests. Do not infer patient identity by taking arbitrary filename fragments.
2. For a historical result, reuse its frozen membership. For a new experiment, choose a patient-level, disease-stratified partition policy and record its seed and any infeasible-class/small-cohort handling. No script here claims to regenerate the historical 80/20 split from aggregate counts.
3. Reserve held-out test patients for final reporting. If model selection needs validation, define a distinct validation partition within the available development patients and record it. Adding one now changes the historical development protocol; it does not retroactively prove a historical run used validation. Do not pass `test` as a validation argument merely because a training command defaults to it.
4. Write one manifest row per acquisition tube, with `patient_id`, `csv_path`, patient-level `label` where applicable, and explicit `tube_type` when supported. Preserve the CSV event labels and model class-map order separately. See [schemas and path semantics](data_format.md).
5. Validate the explicitly chosen sheets using `scripts/validate_manifests.py`. Its default is three disjoint partitions; use `--splits train test` only for a deliberate two-partition protocol. A missing requested sheet is an error, never a reason to substitute the first sheet.
6. Use identical cohort eligibility, tube order, sampling policy, marker vocabulary, class mapping, feature artifacts and model checkpoint when comparing training-time and standalone evaluation. Keep PACTransformer train/validation/test embedding payloads separate and retain their original row-index arrays.
7. Store exact commands, source tag/commit, environment, preparation records, manifest hashes, learned preprocessing artifacts, model-selection records and aggregate results with the authorized run. Inspect a local validation report before sharing it; aggregate counts and column names may still need institutional review.

## Restricted data and public examples

This repository's code license does not grant rights to clinical or third-party datasets. Request access through each data owner's approved process; this release does not promise that access will be granted. Clinical raw data, row-level predictions, real patient IDs, private filesystem paths and patient-level manifests remain outside the release.

Public examples and tests use generated identifiers and values only. They test shape/IO contracts and leakage guards, not clinical accuracy, the reported 80/20 membership, or acquisition-specific compensation and gating.
