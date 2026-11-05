# Repaired issues and compatibility

The v1.0.0 source repairs the eight implementation issues identified in the
release candidate. Their sixteen original regression cases now run as ordinary
assertions, with additional boundary coverage. All fixtures are synthetic;
clinical experiments and manuscript scores have not been rerun.

## Repairs in v1.0.0

| ID | Affected code | Corrected behavior |
| --- | --- | --- |
| R001 | `src/cell_level/baselines/_eval.py` | LR evaluation decodes a single binary coefficient row with the sklearn decision threshold and maps predicted columns through the saved `classes`, including noncontiguous classes. |
| R002 | `src/common/patient_baselines.py` | A missing requested split raises. A single-sheet workbook may supply partitions through a valid split column; it cannot silently substitute training rows for test rows. |
| R003 | FlowCat and pooled/per-tube GMM models | Saved seeds use deterministic patient/tube identities rather than Python `hash()`. Per-file sampling does not depend on the input path, including after moving or renaming a file. |
| R004 | `src/common/patient_paper.py` and patient baseline callers | Explicit `tube_type` values are preserved, with filename inference for missing values. Numeric IDs stay consistent across sheets with missing cells; textual IDs retain leading zeros. Mixed explicit and numeric tube identities can be ordered together. |
| R005 | `src/cell_level/baselines/set_transformer/helpers.py` | Cache identity includes source content, ordered markers, class mapping, sampling limit and seed. Changed inputs/configuration and damaged arrays trigger fresh cache construction. |
| R006 | `src/cell_level/pactransformer/eval.py` | AP/PR-AUC point estimates use all valid test events. Capped bootstrap pools preserve rounded class proportions; caps too small to retain both classes return NaN bounds. Capped intervals remain approximations; use the complete cohort for full-data bootstrap intervals. |
| R007 | `src/cell_level/baselines/_train.py` | Explicit DeepCyTOF epochs do not inspect an unrelated reference workbook. Automatic step alignment retains its reference-count dependency. |
| R008 | CTransformer evaluation and Set Transformer manifest readers | Both apply the shared `FCM_ML_PATH_MAP` remapping before file existence checks. |

## Running the regression checks

```bash
python -m pytest tests/core/test_release_regressions.py -q
python -m pytest -q
```

The original sixteen cases have no expected-failure markers. The complete count
and environment are recorded in [validation](validation.md). Checks use temporary
CSV/XLSX files, tiny LR fits, bounded SOM calculations, and mocked training where
needed; they do not require clinical files or GPUs.

## Compatibility with historical outputs

These repairs intentionally change affected predictions, feature sampling,
cache reuse, panel identities and AP estimates. Stable sampling is reproducible
under the corrected implementation; it does not reproduce a historical process
whose seed depended on Python hash randomization or absolute file paths. Keep
old artifacts and their source revision when comparing historical runs.

A missing partition now fails early instead of reading the first sheet. Provide
explicit nonempty partitions and validate patient-ID disjointness before training.
Old Set Transformer cache entries are ignored by the corrected cache identity.
Compare clinical results only with the original authorized data, selected
checkpoints and recorded protocols; synthetic regressions do not establish the
effect of these defects on a manuscript table.

## Protocol choices that require explicit run records

These are not proof that the manuscript used an incorrect protocol:

- CA-MIL and TACA-MIL training default `val_sheet` to `test`; use an independent
  validation sheet for checkpoint selection and record the actual command.
- CA-MIL and TACA-MIL have different default evaluation filtering/cell-limit
  policies. Specify the intended policies and report the evaluated cohort rather
  than assuming the defaults are equivalent.
- Patient `count` filtering selects the modal tube count; `signature` filtering
  can depend on manifest row order. Record the selected patient IDs privately and
  release nonidentifying cohort counts and rules when permitted.
- Cell-level labels must be covered by the selected mapping; unknown labels can
  currently become class zero. MAE supervised evaluation expects integer-coded
  labels as documented, not arbitrary textual diagnosis names.

Historical final-run checkpoints, predictions, and private split manifests are
needed to determine whether a limitation affected a particular reported table.
Synthetic failure demonstrations alone cannot establish that manuscript results
are incorrect.
