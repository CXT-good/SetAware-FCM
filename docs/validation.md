# Software validation

## Runtime checked on 2026-11-05

The final v1.0.0 source is checked in an isolated overlay around CPython
**3.9.25**, with PyTorch **2.8.0** compiled for CUDA **12.8**, NumPy
**2.0.2**, pandas **2.3.1**, scikit-learn **1.6.1** and Matplotlib
**3.9.4**. Execution uses CPU; no CUDA device is available to this process.
The shared scientific environment is not modified.

All **49 active non-PyTorch release-version pins** match installed package
metadata, and `pip check` reports no broken requirements. PyTorch's runtime
reports `2.8.0+cu128` and CUDA `12.8`, while its installed distribution metadata
reports `2.8.0`. This available runtime is not a fresh installation/verification
of the official `+cu128` or `+cpu` wheels. The fixed requirements explicitly
select those official builds for readers; see [environment](environment.md).

## Regression results

The complete suite reports **193 passed, 0 failed, 0 expected failures**.
Three existing PyTorch warnings concern disabled nested-tensor optimization
when `norm_first=True`; they do not report a test failure.

All **16 original regression cases** for R001–R008 were reproduced failing
before their repairs and now run without expected-failure markers. Added
boundary cases cover split-column precedence and single-sheet train/test
isolation, numeric/text tube identity across sheets, nested per-tube sampling,
LR score ties and saved class IDs, cache content/configuration changes,
damaged cache arrays, path remapping, explicit/automatic DeepCyTOF epochs,
and small/imbalanced bootstrap pools. Independent source review found and
verified the additional split and tube-identity repairs before completion.

PAC AP/PR-AUC point estimates use the complete finite-score test set.
Bootstrap pools capped below the full cohort preserve rounded class
proportions; their intervals are approximations and may differ from a full
bootstrap. If a cap rounds either class count to zero, bounds are NaN while
the full-data point estimate remains available. Use the full cohort size
for `--bootstrap_max_n` when full-data bootstrap intervals are required.

Other checks cover model interfaces, patient-checkpoint restoration without
the original pretrained path, saved baseline artifacts, synthetic partition
disjointness, manifest validation, recipe CLI parsing, fixed-version metadata,
CI/citation integration and source scanning. All **21 paper recipes** have
their actual CLI arguments parsed without clinical data or training.

## Packaging and synthetic checks

Source and wheel distributions build successfully with `python -m build
--no-isolation`. Archive inspection finds no clinical CSVs, patient workbooks,
model binaries, caches or private logs. The only CSVs are the sanitized source
map and experiment-manifest template. The source archive includes requirements,
recipes/evidence, guides, tests, CI, citation, MIT license and third-party notices.

All **93 Python package modules** import from the isolated installed wheel
under the runtime above. The five-model synthetic CPU demo passes against
that installed wheel. Synthetic patient and cell manifests both validate,
including train/validation/test patient disjointness. No clinical training,
checkpoint selection or manuscript metric reproduction is implied.

Compared with public source commit
`116fcc3846b748d54c4ddc4f6d4bcfb53879572b`, **11 source files** contain defect
repairs and **83 source files** retain their published bytes. The **151-entry**
`docs/source_map.csv` records current release hashes and original relative
identities. Historical recipes and their evidence have not been substituted
with new protocols. See [repaired issues](known_issues.md) for behavior changes.

MIT adoption and publication including the repairs were authorized on
2026-11-05. These software checks do not establish that a remote tag, GitHub
release, package-index upload or public CI run has occurred. Confirm external
publication separately on the repository.

## Repeat the checks

Install the appropriate fixed environment first, then run from the repository
root:

```bash
python -m pip check
python scripts/check_release.py
OMP_NUM_THREADS=1 python -m pytest -q
python examples/synthetic_demo.py
python examples/make_synthetic_data.py --output-dir outputs/synthetic
python scripts/validate_manifests.py \
  --manifest outputs/synthetic/patient_split.xlsx --level patient
python scripts/validate_manifests.py \
  --manifest outputs/synthetic/synthetic_dataset_split.xlsx --level cell \
  --class-mapping outputs/synthetic/class_mapping.json
python -m build --no-isolation
```

Set `NUMBA_CACHE_DIR` and `MPLCONFIGDIR` to writable directories if needed.
Fixtures are synthetic and temporary. Keep clinical identifiers and local
manifest paths private when inspecting or sharing reports.

## Outside this validation

- Distributed MAE pretraining, CUDA/NCCL training and GPU timings.
- Fresh official CUDA/CPU wheel installation or a public CI result.
- Recovery of missing historical preprocessing, split membership, selected
  checkpoints or baseline artifacts; see [paper configurations](paper_runs.md).
- Clinical experiments, manuscript scores or the impact of the repairs on
  a specific historical result table.
