# Data access and local layout

GAHIB's public package keeps data access explicit. Benchmark and robustness
runners read local `.h5ad` files, but the repository should not encode private
workstation paths, private correspondence, or unpublished data locations.

## Required local input format

- Input files are AnnData `.h5ad` files.
- Benchmark filenames should match the dataset identifiers listed in
  `experiments/exp_utils.py::SELECTED_DATASETS`.
- The public-safe benchmark metadata manifest is documented in
  `docs/DATASET_MANIFEST.md`; the machine-readable CSV is
  `docs/DATASET_MANIFEST.csv`.
- Each file should contain raw counts in `adata.X` or a count-compatible layer;
  `experiments/exp_utils.py::load_and_preprocess` saves raw counts to
  `layers["counts"]`, normalizes to 10,000 counts per cell, applies `log1p`,
  selects 2,000 highly variable genes, and caps each dataset at 3,000 cells.

## Configure dataset locations

Set `GAHIB_DATASET_DIRS` to a colon-separated list of directories containing
public or locally authorized `.h5ad` files:

```bash
export GAHIB_DATASET_DIRS="/path/to/CancerDatasets:/path/to/DevelopmentDatasets"
python experiments/run_ablation.py
```

The `/path/to/...` entries above are examples only. Do not commit private paths,
access tokens, private correspondence, or unpublished raw data locations.

## Included small data

`data/pbmc3k_processed.h5ad` is a small local example artifact for smoke checks
and examples. If `GAHIB_DATASET_DIRS` is unset, dataset discovery searches only
this repo-local `data/` directory. The full 53-dataset benchmark is intentionally
external and must be supplied through `GAHIB_DATASET_DIRS`.

## Optional local MSigDB gene-set files

`experiments/run_go_enrichment.py` uses local MSigDB GMT files for offline GO
Biological Process enrichment. Configure these paths locally instead of editing
the script:

```bash
export GAHIB_MSIGDB_DIR="/path/to/msigdb"
# or override files individually:
export GAHIB_GMT_HUMAN="/path/to/c5.go.bp.v2024.1.Hs.symbols.gmt"
export GAHIB_GMT_MOUSE="/path/to/m5.go.bp.v2024.1.Mm.symbols.gmt"
```

For a public archive, include only redistribution-permitted gene-set files or
document where users can obtain them from the original provider.

## Public release checklist

Before archiving or publishing a release, verify that the public repository has:

1. public accession or download notes for every benchmark cohort that can be
   redistributed;
2. a statement for any cohort that requires users to obtain data from the
   original provider;
3. no absolute private paths such as `/home/...`, `/Users/...`, or
   workstation-specific download-directory locations in public docs/scripts;
4. no private correspondence or response drafts in public package docs.
