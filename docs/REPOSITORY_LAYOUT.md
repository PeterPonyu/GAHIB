# Public repository layout

Date: 2026-05-16

`GAHIB/` is the public code and reproducibility repository. It should remain
safe to publish without local workstation paths, non-redistributable datasets,
submission drafts, generated PDFs, or unpublished evidence tables.

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
- Keep non-redistributable input datasets, benchmark outputs
  (`GAHIB_results/`), generated figures, manuscript PDFs, and submission bundles
  out of the public release.
- Historical manuscript or revision folders may exist in local checkouts for
  compatibility with older scripts, but public package docs and runnable examples
  should not depend on them.

## Path safety

Public scripts still use `GAHIB_results/{study}/` as their output root for
backward compatibility. Do not move that output path unless the scripts are first
updated to accept an environment-configurable results root and the public tests
are rerun.

Use repo-relative paths and documented environment variables in public examples.
Do not commit absolute workstation paths or unpublished data locations.
