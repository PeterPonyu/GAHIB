# Public benchmark dataset manifest

`DATASET_MANIFEST.csv` is a public-safe metadata manifest for the 53 benchmark
cohorts used by the GAHIB benchmark package. It contains metadata derived from
available project manifests only; it does not include raw data, private
workstation paths, private correspondence, or unpublished local download
locations.

## Scope

- Rows: 53 datasets.
- Public identifiers: GEO-style accessions are inferred when present in dataset
  identifiers; legacy short names are flagged as unresolved in the existing
  manifest.
- Available fields: dataset ID, domain, cell/gene counts, local file-size proxy,
  label-column availability, and inferred species/tissue-family strata.
- Explicitly unavailable fields: `platform`, `cluster_count`, `units`, and
  `qc_normalization` are marked `not_available_in_existing_manifest` because the existing revision
  manifests do not expose source-level values.

## How to use with public code

Place authorized `.h5ad` files in directories listed by `GAHIB_DATASET_DIRS`.
Filenames should match `benchmark_filename` in the CSV. The public package does
not redistribute the full benchmark raw data.

## Claim guardrail

This manifest supports bounded provenance/stratification statements. It should
not be cited as evidence that all source-level platform, unit, cluster, or
QC/normalization metadata were manually audited.
