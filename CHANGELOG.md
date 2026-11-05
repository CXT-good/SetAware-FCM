# Changelog

## 1.0.0 — 2026-11-05

- Retain the flat, model-centred source layout from the earlier public
  source snapshot, commit `116fcc3846b748d54c4ddc4f6d4bcfb53879572b`.
- Add version-fixed Python 3.9.25 / PyTorch 2.8.0 CUDA 12.8 and CPU profiles,
  package metadata, source-distribution manifests and CI.
- Add five manuscript-derived recipes and sixteen recovered historical
  patient train/eval recipes, with sanitized source hashes and explicit
  unresolved evidence rather than invented final-run settings.
- Document prepared-CSV preprocessing, patient-level partition rules and
  method-specific feature handling; add a strict local manifest validator.
- Publish the regression suite, synthetic examples, collective software
  citation metadata and third-party attention-code notice.
- Adopt the MIT project license, authorized on 2026-11-05, retaining third-party notices.
- Repair binary/noncontiguous LR class decoding, missing patient-split rejection,
  deterministic FlowCat/GMM sampling, and explicit patient tube identity.
- Invalidate Set Transformer caches when input content or configuration changes;
  apply shared manifest path remapping in CTransformer and Set Transformer.
- Compute PAC AP point estimates on the complete valid test set and remove
  positive-class enrichment from capped bootstrap pools. Return undefined
  interval bounds for caps too small to represent both classes; capped
  intervals remain approximate.
- Remove unrelated reference-workbook access for explicit DeepCyTOF epochs.
- Convert all sixteen known-defect regressions into ordinary tests and add
  boundary coverage. Corrected behavior can change affected historical outputs;
  clinical results have not been rerun.
