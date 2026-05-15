# Reproducibility and quickstart commands

This page collects public-safe commands for installing GAHIB, running smoke
checks, and reproducing benchmark tables from local `.h5ad` inputs. Commands are
repo-relative and avoid private workstation paths.

## 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
# Optional graph encoders:
python -m pip install -e ".[graph]"
```

## 2. Fast public smoke checks

```bash
python -c "from gahib import GAHIB; print(GAHIB.__name__)"
python -m compileall gahib experiments
pytest -q tests/test_models.py::TestBaseVAE::test_instantiation
pytest -q tests/test_models.py::TestGraphEncoder::test_graph_decoder_accepts_float64_edge_weights
```

Run the full synthetic training suite when dependencies and compute are
available:

```bash
pytest -q
```

The full test suite trains small synthetic models and may take longer than an
import-only smoke check.

## 3. Configure benchmark data

```bash
export GAHIB_DATASET_DIRS="/path/to/CancerDatasets:/path/to/DevelopmentDatasets"
```

`GAHIB_DATASET_DIRS` must point to directories containing `.h5ad` files whose
basenames match the identifiers in `experiments/exp_utils.py`. If it is unset,
dataset discovery searches only the repo-local `data/` directory. Example paths
are placeholders; replace them locally and do not commit private paths.

## 4. Run public benchmark scripts

Single study examples:

```bash
python experiments/run_ablation.py
python experiments/run_hyperparam_sensitivity.py
python experiments/run_seed_robustness.py
python experiments/run_computational_cost.py
```

Grouped runners:

```bash
bash experiments/run_parallel_group_A.sh
bash experiments/run_parallel_group_B.sh
bash experiments/run_all_expanded.sh
```

Outputs are written under `GAHIB_results/`. Generated tables, logs, and figures
should be reviewed before publication and archived with accession/data-source
notes from `docs/DATA.md`.

For GO-enrichment runs, configure local MSigDB GMT files through environment
variables instead of editing scripts:

```bash
export GAHIB_MSIGDB_DIR="/path/to/msigdb"
# or: export GAHIB_GMT_HUMAN="/path/to/c5.go.bp.v2024.1.Hs.symbols.gmt"
#     export GAHIB_GMT_MOUSE="/path/to/m5.go.bp.v2024.1.Mm.symbols.gmt"
python experiments/run_go_enrichment.py
```

## 5. Public-path audit

Run this check before tagging a public release:

```bash
rg -n --hidden \
  -g '!*.pdf' -g '!*.png' -g '!*.jpg' -g '!.git/**' -g '!.venv/**' \
  '/ho[m]e/|/Us[e]rs/|Downloa[d]s|passwor[d]|toke[n]|secre[t]|confiden[t]ial' .
```

Expected findings should be limited to documented example placeholders or the
public scope note under `revision/99_notes/`. Replace any executable private
paths in scripts with repo-relative paths or environment variables.
