<div align="center">
  <a href="https://peterponyu.github.io/">
    <img src="https://peterponyu.github.io/assets/badges/GAHIB.svg" width="64" alt="ZF Lab · GAHIB">
  </a>
</div>

# GAHIB

[![PyPI](https://img.shields.io/pypi/v/gahib.svg)](https://pypi.org/project/gahib/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24268692.svg)](https://doi.org/10.5281/zenodo.24268692)

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

## Install

```bash
pip install gahib
# graph encoders, scanpy and scVI helpers
pip install "gahib[all]"
```

For development, clone the repository and run `pip install -e ".[all]"`.

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
distance becomes degenerate. Set `irecon` above zero whenever `lorentz`
is above zero, as in the usage example; the model does not check this.

## Main outputs

| Call                   | Returns                                                                                |
| ---------------------- | -------------------------------------------------------------------------------------- |
| `model.get_latent()`   | latent code sampled from each cell's posterior, shape `(n_cells, latent_dim)`          |
| `model.get_centroid()` | posterior mean of each cell's latent code (no sampling), shape `(n_cells, latent_dim)` |

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

## Citation

If you use GAHIB, please cite the article:

> Fu Z, Fu J, Wang X, Liu Y, Ran T. GAHIB: graph attention VAE with a hyperbolic information bottleneck for biologically structured single-cell representations. *Frontiers in Genetics*. 2026;17:1863100. https://doi.org/10.3389/fgene.2026.1863100

```bibtex
@article{fu2026gahib,
  author  = {Fu, Zeyu and Fu, Jiawei and Wang, Xiaoxia and Liu, Yiyao and Ran, Tianfei},
  title   = {GAHIB: graph attention VAE with a hyperbolic information bottleneck for biologically structured single-cell representations},
  journal = {Frontiers in Genetics},
  volume  = {17},
  pages   = {1863100},
  year    = {2026},
  doi     = {10.3389/fgene.2026.1863100}
}
```

GitHub's "Cite this repository" button reads the same record from `CITATION.cff`.

## License

MIT. See `LICENSE`.
