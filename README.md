<div align="center">
  <a href="https://peterponyu.github.io/">
    <img src="https://peterponyu.github.io/assets/badges/GAHIB.svg" width="64" alt="ZF Lab · GAHIB">
  </a>
</div>

# GAHIB

A PyTorch implementation of a graph-attention variational autoencoder with
an information bottleneck and a Lorentz hyperbolic geometry loss, applied
to single-cell RNA-seq latent representation learning.

## What it does

Given a `scanpy` `AnnData` with raw counts, GAHIB fits a VAE whose encoder
can be an MLP, a Transformer, or a graph neural network (GAT, GCN,
GraphSAGE, ChebConv, TAG, GraphTransformer, ARMA). The latent code is
shaped by three losses:

- a reconstruction term (NB, ZINB, Poisson, or ZIP),
- an information-bottleneck term that compresses the latent into a 2D
  manifold coordinate, and
- a Lorentz-hyperbolic term that anchors the manifold coordinate on the
  hyperboloid so radial distance encodes hierarchy.

Optional SDE and PDE modules are provided for trajectory experiments.

## Install

```bash
pip install -e .
# optional extras
pip install -e ".[all]"
pip install -e ".[dev]"
```

Python 3.9 or later. Full dependency list in `pyproject.toml`.

## Usage

```python
import scanpy as sc
from gahib import GAHIB

adata = sc.read_h5ad("data/human_cd34_bone_marrow.h5ad")

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
Z = model.get_latent()
```

Set `encoder_type="mlp"` (default) for a plain VAE without the graph
encoder.

## Important constraint

The hyperbolic loss depends on the information-bottleneck target. With
`irecon=0`, the bottleneck coordinate is untrained and the Lorentz
distance becomes degenerate. The model enforces this dependency
internally.

## Main outputs

| Call                  | Returns                               |
| --------------------- | ------------------------------------- |
| `model.get_latent()`  | the learned embedding                 |
| `model.get_time()`    | the trajectory time estimate (if SDE) |
| `model.get_centroid()`| per-cluster latent centroids          |

## Experiment runners

The `experiments/` directory contains reproducible runners for the
benchmark and robustness studies. Each runner takes its dataset paths
from `GAHIB_DATASET_DIRS` and writes results to
`GAHIB_results/{study}/`.

## Repository layout

```text
gahib/       package source (core model, metrics, interpretation)
experiments/ benchmark and robustness runners
tests/       pytest suite
```

## Tests

```bash
pytest
```

## License

MIT. See `LICENSE`.
