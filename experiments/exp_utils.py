"""
Shared utilities for GAHIB experiment scripts.
============================================
Follows the training-pipeline skill Phase 2 preprocessing exactly:
  1. Save raw counts in layers['counts']
  2. Normalize (target_sum=1e4) + log1p
  3. Select highly variable genes (n_top_genes=2000)
  4. Subsample to max_cells=3000
  5. Subset to (subsampled cells) × (HVG genes) and .copy()

The resulting adata1 is used by ALL models:
  - GAHIB variants:  GAHIB(adata1, layer='counts', ...)
  - External models: X = adata1.X.toarray()  (normalized HVG)
  - Labels:          get_labels(adata1)
"""

import sys, os, logging
import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)
import scipy.sparse as sp

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from gahib.metrics import compute_all_metrics

# ── Constants ──
MAX_CELLS = 3000
N_HVG = 2000
SEED = 42

from experiments.benchmark_config import (
    SELECTED_DATASETS,
    get_configured_dataset_dirs,
    match_benchmark_datasets,
)


def discover_datasets():
    """Find h5ad files and filter to selected benchmark datasets.

    Dataset directories are configurable via the GAHIB_DATASET_DIRS environment
    variable (colon-separated list of paths). If unset, search only the
    repo-local data/ directory so public scripts do not encode private
    workstation defaults.
    """
    search_dirs, env_was_set = get_configured_dataset_dirs(project_root=PROJECT_ROOT)
    if not env_was_set:
        print(
            "GAHIB_DATASET_DIRS is not set; searching repo-local data/ only. "
            "Set GAHIB_DATASET_DIRS for the full benchmark."
        )

    matched, missing = match_benchmark_datasets(search_dirs)
    for name in missing:
        print(f"⚠ Dataset not found: {name}")
    return [matched[name] for name in SELECTED_DATASETS if name in matched]


def get_labels(adata, resolution=1.0):
    """Compute unsupervised reference labels via Leiden clustering.

    All benchmarking uses Leiden on preprocessed data as the reference
    partition.  Ground-truth cell type annotations are never used,
    ensuring fully unsupervised evaluation.

    Parameters
    ----------
    adata : AnnData
        Preprocessed data (normalized, log-transformed, HVG-selected).
    resolution : float
        Leiden resolution parameter.  Default 1.0.

    Returns
    -------
    labels : ndarray of str
        Leiden cluster assignments.
    n_clusters : int
        Number of Leiden clusters found.
    """
    leiden_key = f'leiden_{resolution}'
    if leiden_key not in adata.obs.columns:
        if 'neighbors' not in adata.uns:
            use_rep = 'X_pca' if 'X_pca' in adata.obsm else None
            sc.pp.neighbors(adata, use_rep=use_rep)
        sc.tl.leiden(adata, resolution=resolution, key_added=leiden_key)
    labels = adata.obs[leiden_key].values.astype(str)
    n_clusters = len(np.unique(labels))
    logger.info("  Leiden (res=%.1f): %d clusters", resolution, n_clusters)
    return labels, n_clusters


def load_and_preprocess(filepath):
    """Load h5ad and apply the MANDATORY preprocessing pipeline.

    Returns
    -------
    adata1 : AnnData
        Preprocessed data with:
        - adata1.X = normalized log-transformed HVG expression
        - adata1.layers['counts'] = raw integer counts (HVG subset)
        - adata1.n_obs <= MAX_CELLS (3000)
        - adata1.n_vars = N_HVG (2000)
    """
    adata = sc.read_h5ad(filepath)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # 1. Ensure sparse
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    # 2. Save raw counts BEFORE normalization
    adata.layers['counts'] = adata.X.copy()

    # 3. Normalize + log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 4. Select highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)

    # 5. Subsample cells to fixed size
    rng = np.random.default_rng(SEED)
    if adata.shape[0] > MAX_CELLS:
        idxs = rng.choice(adata.shape[0], MAX_CELLS, replace=False)
    else:
        idxs = rng.permutation(adata.shape[0])

    # 6. Subset to (subsampled cells) x (HVG genes) and COPY
    adata1 = adata[idxs, adata.var['highly_variable']].copy()

    print(f"  Preprocessed: {adata.n_obs} cells -> {adata1.n_obs} cells, "
          f"{adata.n_vars} genes -> {adata1.n_vars} HVGs")

    return adata1


def get_dense_X(adata1):
    """Get dense normalized HVG matrix from preprocessed adata1.
    Use this as input for external models (GM-VAE, etc.)."""
    X = adata1.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def evaluate_latent(latent, labels, dre_k=15):
    """Compute all metrics for a latent embedding."""
    # Encode string labels to integers (compute_all_metrics expects int)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels_int = le.fit_transform(np.asarray(labels).astype(str))
    raw = compute_all_metrics(latent, labels_int, dre_k=dre_k)
    return {k: v for k, v in raw.items()
            if not k.startswith('_') and np.isscalar(v)}


def get_done_datasets(tables_dir, prefix):
    """Check which datasets have already been processed (for resume)."""
    done = set()
    for f in glob.glob(os.path.join(tables_dir, f'{prefix}_*_df.csv')):
        name = os.path.basename(f).replace(f'{prefix}_', '').replace('_df.csv', '')
        done.add(name)
    return done
