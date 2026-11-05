# Paper release scope

## Included materials

The release retains the flat `src/` layout for MAE, CTransformer,
PACTransformer, CA-MIL, TACA-MIL and their baselines. Supporting materials
include fixed software versions, installation metadata, synthetic examples,
regression tests, CI, prepared-data validation, preprocessing/split guidance,
and paper recipes with explicit evidence and recovery status.

See [environment](environment.md), [paper configurations](paper_runs.md),
[preprocessing](preprocessing.md), [known issues](known_issues.md), and the
[validation record](validation.md).

## Excluded materials

Clinical events, real patient identifiers, original patient split workbooks,
trained clinical checkpoints, row-level predictions, generated research
figures/tables, private execution logs, local caches and manuscript history
are not distributed. The aggregate class mapping is not a patient-membership
file. Authorized access to external datasets and models must be arranged
separately; code licensing does not grant those rights.

## What the paper version identifies

A paper-specific tag identifies the source and supporting release files. It
does not retroactively establish the original environment binary, recover
missing partition membership, replace an unavailable selected checkpoint, or
verify clinical metrics. The `clinical_reproduction_verified` fields remain
false until the corresponding clinical comparison has actually been run.

Historical recipes preserve recorded checkpoint selection, including use of
the test sheet. Prospective examples with a separate validation sheet are a
different protocol. Recovered effective evaluation settings take precedence
over nullable historical CLI options or current constructor defaults.

## Citation and provenance

`CITATION.cff` identifies the software with the collective name
"SetAware-FCM contributors". It does not invent personal manuscript authors
or a publication DOI. Add the final article citation when the authors provide
verified bibliographic details.

`docs/source_map.csv` records original relative source identities and current
release-file hashes. Recipe evidence includes hashes of inspected source
records, not the records' protected patient rows. Adapted attention-code
notices are retained in `THIRD_PARTY_NOTICES.md`. The project uses the MIT
`LICENSE` naming Intelligent-Medical-Research-Group. Adoption of MIT and the
scope including R001–R008 repairs were explicitly authorized on 2026-11-05.

## Versioning and release checks

The package version is **1.0.0** and the paper tag is **v1.0.0**. Publication
requires review of the final package tree, passing tests and archive checks,
and recording the exact commit. Existing tags must not be reassigned to a
different commit. Local package preparation does not establish that a remote
tag or release exists; confirm those external actions on the repository.

The sixteen regression cases for R001–R008 are ordinary passing assertions
after repair. The validation summary reports the complete suite and any
remaining expected failures separately. Historical recipes and their
recorded protocol remain intact; repairs do not rerun clinical experiments.

Release notes should link the exact commit, identify material known issues,
state the tested environment, and distinguish configuration recovery from
clinical reproduction. A source archive and package distributions must not
contain data, weights, private run logs or expanded local filesystem paths.
