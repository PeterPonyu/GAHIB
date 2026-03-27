"""
GAHIB Core: Graph Attention Hyperbolic Information Bottleneck
=============================================================

Integrates:
- GAHIB: Hyperbolic VAE with Lorentz geometry, information bottleneck,
  and count-based likelihoods.
- CCVGAE: Graph Attention Network encoders/decoders, graph structure learning,
  subgraph sampling, and centroid inference.

Encoder options: 'mlp', 'transformer', 'graph' (GAT, GCN, ChebConv, SAGE, etc.)
Decoder options: 'mlp' (with NB/ZINB/Poisson/ZIP likelihoods), 'graph'
Regularization: beta-VAE, DIP-VAE, beta-TC-VAE, InfoVAE, Lorentz/Euclidean manifold
"""

from .agent import GAHIB

__all__ = ["GAHIB"]

__version__ = "2.0.0"
