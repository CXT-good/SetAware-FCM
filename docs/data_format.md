# Data formats

## Event CSVs

Each row is a cell; each CSV is one acquisition tube. Marker columns must match the case-sensitive names in `dataio.marker_mapping.MARKER_LIST`; available markers can differ between tubes. Cell-supervised workflows require the event label column (`LABEL_COLUMN`, normally `label`). A patient's diagnosis in the manifest is different from an event's cell-population label in a CSV.

The package consumes prepared numeric event tables. It does not implement a universal FCS compensation, transformation, or gating procedure. The [preprocessing and split protocol](preprocessing.md) separates external preparation from loader behavior and describes which historical details are not recoverable from this release.

Use finite numeric marker values and explicit, nonempty labels. Some loaders coerce malformed numbers to zero or map unknown event labels to class 0; these are historical implementation behaviors, not recommended data-cleaning rules. The validator rejects those ambiguous inputs before training. Do not rename marker columns or reorder the global token vocabulary after training a checkpoint.

`examples/make_synthetic_data.py` writes artificial CSVs using the packaged vocabulary. The synthetic data are suitable for interface checks, not performance comparisons.

## Cell split workbook

Supply an XLSX workbook with `train` and `test` sheets and a `csv_path` column. A separate `validation` sheet can be included for protocols that use it. Use the exact sheet names requested by each command and an absolute workbook path. For audited manifests, also supply a stable pseudonymous `patient_id` for every tube; cell loaders can run without it, but patient-disjointness cannot be established from `csv_path` alone.

Retain patient identifiers locally when constructing splits so all cells and tubes from a patient remain in the same partition. Shared/public manifests should use permitted pseudonymous identifiers and portable paths. Do not recreate a historical split by choosing an arbitrary new seed.

The class mapping has this shape:

```json
{
  "version": 1,
  "datasets": {
    "dataset_name": {
      "class_list": ["class_0", "class_1"],
      "class_counts": [100, 120]
    }
  }
}
```

Counts are an ordered numeric list aligned with `class_list`. `synthetic_dataset_split.xlsx`, for example, selects the `synthetic` dataset key. See `dataio.class_lists.dataset_key_from_split_xlsx` for the full naming rules. Preserve the historical mapping for historical checkpoints. For new data, explicitly choose the count source rather than relying on the builder's default `total` sheet:

```bash
python -m cell_level.ctransformer.build_class_mappings \
  --data_dir /example/prepared-data --sheet train \
  --output_path /example/prepared-data/class_mapping.json
```

`FCM_ML_CLASS_MAPPING_PATH` can select a replacement for the bundled mapping. The bundled research mapping contains aggregate class labels/counts, not patient manifests. The synthetic generator produces `synthetic_dataset_split.xlsx`; replace the generic `cell_split.xlsx` path in an example recipe with that filename when using the generated data. Changing the count source can change class weights; this is not an assertion that the command recreates every historical mapping.

## Patient split workbook

Each row describes one acquisition tube. Required columns are `patient_id`, `csv_path`, and `label`; `tube_type` is optional but recommended. A patient can have multiple rows, all with the same patient label. The CSV contains event-marker values, not previously exported model embeddings.

| patient_id | csv_path | label | tube_type |
| --- | --- | --- | --- |
| synthetic_000 | events/synthetic_000_tube_0.csv | class_0 | panel_0 |
| synthetic_000 | events/synthetic_000_tube_1.csv | class_0 | panel_1 |

`filter_mode=count` retains the modal number of tubes; `signature` retains the modal tube signature; `none` keeps all eligible patient records. `max_cells_per_tube` and sampling settings change the evaluated input and must be recorded. Training and evaluation defaults are not interchangeable.

Keep row and tube order stable: some sampling seeds depend on patient/tube indices. Supply nonempty explicit `tube_type` values for every row. Patient baseline tables preserve explicit IDs and infer missing/blank entries from `tube_0`, `tube_1`, etc. filenames. CA-MIL/TACA-MIL infer from the CSV parent/stem only when the entire column is absent; they do not fill individual blank entries. Verify the actual channel selection for the chosen model. CA-MIL/TACA-MIL select marker-vocabulary columns, whereas some baselines select numeric columns more broadly. For those baselines, remove numeric identifiers, event labels and non-feature metadata from a separate feature-only copy of each input CSV and record the selected column list.

## Validate before training

Run from the repository root after installing the package as described in the README. The examples below use only artificial data:

```bash
python examples/make_synthetic_data.py --output-dir outputs/synthetic
python scripts/validate_manifests.py \
  --manifest outputs/synthetic/patient_split.xlsx --level patient \
  --report outputs/validation/patient_manifest.json
python scripts/validate_manifests.py \
  --manifest outputs/synthetic/synthetic_dataset_split.xlsx --level cell \
  --class-mapping outputs/synthetic/class_mapping.json \
  --report outputs/validation/cell_manifest.json
```

The validator requires `patient_id` and `csv_path` for both levels, plus manifest `label` for patient tasks and an explicit class mapping for cell tasks. By default it requires all three `train`, `validation`, and `test` partitions. For a deliberately two-partition historical workbook, explicitly pass `--splits train test`; this validates the two partitions and does not invent a validation cohort. Additional workbook sheets such as `total` are not evaluated unless supported and selected; `total` is never a held-out partition.

The helper also accepts a CSV manifest with the same columns and an explicit `split` column whose values are `train`, `validation`, or `test`. Model commands still consume their documented XLSX format; validator CSV support does not add CSV-manifest support to those models.

Checks cover required/empty fields, explicitly requested splits, duplicate resolved tube paths, patient overlap across selected splits, inconsistent patient diagnoses, unseen held-out diagnoses, existing CSVs, at least one recognized marker, finite float32 marker values, and nonempty/mapped event labels for cell tasks. Event tables are read in chunks (`--chunksize`, default 100,000). Failures return exit code 2; success prints aggregate counts and can save a local JSON report. The report does not contain patient identifiers or event measurements, but it may list ignored column names.

Unknown event columns are reported as `ignored_event_columns`, matching the marker-vocabulary loaders; the standard `Time`, `event_ID`, and configured label columns are ignored without warnings. Use `--strict-marker-columns` to reject other unknown names and `--ignore-column metadata_name` to declare known metadata. This can catch misspelled marker names without forcing all legacy metadata to be model features. This check does not certify the broader feature-selection behavior of every baseline.

Validation does not reconstruct an unavailable historical split, verify disease stratification when no disease category is supplied, detect the same patient recorded under different pseudonyms, or detect copied event content at different paths. It also does not validate FCS compensation/gates or prove a workbook corresponds to a published result. Keep local patient-identity reconciliation and run provenance separately.

## PACTransformer payloads

CTransformer embedding export writes three matching directories:

```text
payload/
├── embeddings/tube_a_embeddings.npy  # [cells, embedding_dimension]
├── labels/tube_a_labels.npy          # [cells]
└── xuhao/tube_a_xuhao.npy            # [cells], original row indices
```

Keep filenames, row counts, ordering, labels, and original row indices aligned. Create separate payloads for train/validation/test; never pass the training cache as a held-out test set.

## Relocate manifest paths

Prefer paths such as `events/synthetic_000_tube_0.csv` in portable workbooks. **Relative event paths resolve against the process working directory, not the workbook's directory**, in the published loaders and validator. Run validation and model commands from the same documented data root, or generate local absolute event paths after relocation. The synthetic generator writes local absolute paths for immediately runnable demonstrations, so regenerate its data after moving it.

If an externally supplied workbook retains an archived prefix, map it explicitly:

```bash
export FCM_ML_PATH_MAP='{"/archive/events":"/example/data/events"}'
```

Core split readers and the validator apply this JSON prefix mapping before checking event paths. The longest complete matching prefix takes precedence. A workbook with correct local paths needs no mapping. The class-mapping builder is a separate utility and does not apply this environment remapping: give it workbooks with already valid local paths.
