# Public repository layout

Date: 2026-05-16

`GAHIB/` is the public code and reproducibility repository. Manuscript drafting,
reviewer-response packages, generated PDFs, and private/local data are managed in
`../GAHIB-assets/`.

## Canonical public tree

```text
GAHIB/
  README.md
  LICENSE
  pyproject.toml
  gahib/          # Python package: model, graph modules, metrics
  experiments/    # public reproducibility runners
  tests/          # public smoke/synthetic tests
  docs/           # public dataset manifest and reproducibility notes
```

## Boundaries

- Keep public-safe dataset metadata in `docs/DATASET_MANIFEST.csv` and
  `docs/DATASET_MANIFEST.md`.
- Keep raw data (`data/`), benchmark outputs (`GAHIB_results/`), generated
  figures, manuscript PDFs, and submission bundles out of the public release.
- Historical paper/revision folders may exist in local checkouts for compatibility
  with old scripts, but the auditable Frontiers revision dossier is now sorted in
  `../GAHIB-assets/submission/`, `../GAHIB-assets/versions/`, and
  `../GAHIB-assets/evidence/`.

## Path safety

Public scripts still use `GAHIB_results/{study}/` as their output root for
backward compatibility. Do not move that output path unless the scripts are first
updated to accept an environment-configurable results root and the public tests
are rerun.
