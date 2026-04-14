# GAHIB

GAHIB is a PyTorch package for learning single-cell latent spaces with a graph-attention variational autoencoder, an information bottleneck, and a hyperbolic geometry loss.
The repository also contains the experiment runners and figure generators used for the paper.

## What the package does

- Learns latent representations from scRNA-seq count matrices.
- Supports MLP, Transformer, and graph encoders, including GAT, GCN, GraphSAGE, ChebConv, TAG, GraphTransformer, ARMA, and related PyTorch Geometric modules.
- Supports NB, ZINB, Poisson, and ZIP reconstruction models.
- Includes optional information-bottleneck, Euclidean, and Lorentz-hyperbolic regularisers.
- Includes optional SDE and PDE modules for trajectory-style experiments.
- Provides experiment scripts, benchmark aggregation, and publication figure generation.

## Installation

```bash
pip install -e .
pip install -e ".[all]"
pip install -e ".[dev]"
```

The repository expects Python 3.9+.
The exact dependency list is in `pyproject.toml`.

## Minimal example

```python
import scanpy as sc
from gahib import GAHIB

adata = sc.read_h5ad("data/BoneMarrow/human_cd34_bone_marrow.h5ad")

model = GAHIB(
    adata,
    layer="counts",
    encoder_type="graph",
    graph_type="GAT",
    latent_dim=10,
    i_dim=2,
    irecon=1.0,
    lorentz=5.0,
)

model.fit(epochs=200, patience=30)
latent = model.get_latent()
```

For a plain VAE without graph encoding, leave `encoder_type` at its default.

## Important constraint

The geometry loss needs the information bottleneck.
If `irecon=0`, the bottleneck target is not trained, so the geometry distance is not meaningful.
The code already enforces this relation.

## Common outputs

- `model.get_latent()` returns the learned embedding.
- `model.get_time()` returns the trajectory-style time estimate when the relevant module is enabled.
- `model.get_centroid()` returns latent centroids.

## Running the paper assets

Paper figures are exposed through `paper/figures/` and are generated from the results under `GAHIB_results/`.
From the repository root:

```bash
python -m gahib.viz.fig_overview
python -m gahib.viz.fig_architecture
python -m gahib.viz.fig_dataset_taxonomy
python -m gahib.viz.run_all_visualizations
python -m gahib.viz.fig_downstream_analysis
python experiments/run_interpretation.py --figures-only
```

To rebuild the manuscript:

```bash
cd paper
make figures
make
```

## Main experiment scripts

- `python experiments/run_ablation.py`
- `python experiments/run_sc_deeplearning_benchmark.py`
- `python experiments/run_classical_benchmark.py`
- `python experiments/run_gmvae_benchmark.py`
- `python experiments/run_disentanglement.py`
- `python experiments/run_encoder_comparison.py`
- `python experiments/run_graph_conv_sweep.py`
- `python experiments/run_latent_dim_ablation.py`
- `python experiments/run_seed_robustness.py`
- `python experiments/run_computational_cost.py`
- `python experiments/run_hyperparam_sensitivity.py`
- `python experiments/run_interpretation.py`

## Repository layout

- `gahib/`: package source code.
- `experiments/`: benchmark and analysis runners.
- `data/`: local datasets used by the experiments.
- `GAHIB_results/`: generated benchmark outputs and figures.
- `paper/`: manuscript source and paper assets.
- `tests/`: test suite.

## Current study scope

The current paper version evaluates 53 scRNA-seq datasets across 11 study tracks: seven comparative benchmarks and four robustness studies.
The manuscript and the detailed results live in `paper/` and `STUDY_REPORT.md`.

## License

See `LICENSE`.
