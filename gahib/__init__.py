"""
GAHIB: Graph Attention Hyperbolic Information Bottleneck
========================================================

A unified deep learning framework for single-cell omics analysis combining:
- Variational Autoencoder (VAE) with count-based likelihoods (NB, ZINB, Poisson, ZIP)
- Lorentz (hyperbolic) geometric regularization for hierarchical structure
- Dual-path information bottleneck for coordinated biological programs
- Graph neural network encoders (GAT, GCN, ChebConv, SAGE, etc.)
- Multi-encoder: MLP, Transformer, and Graph backbones
- Disentanglement: beta-VAE, DIP-VAE, TC-VAE, InfoVAE regularizers
"""

import logging

logging.getLogger("gahib").addHandler(logging.NullHandler())

from .core.agent import GAHIB

__all__ = ["GAHIB"]

__version__ = "2.0.0"
__author__ = "Zeyu Fu"
__project__ = "GAHIB"
__full_name__ = "Graph Attention Hyperbolic Information Bottleneck"
