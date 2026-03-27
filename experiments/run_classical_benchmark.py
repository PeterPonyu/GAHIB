#!/usr/bin/env python3
"""
Experiment 9: Classical Dimensionality Reduction Benchmark
===========================================================
Compares GAHIB against classical (non-deep) dimensionality reduction methods.
These serve as lower-bound baselines demonstrating the advantage of nonlinear
neural approaches for single-cell data.

Baselines (5 classical + 1 proposed):
  1. PCA            — Principal Component Analysis (linear)
  2. ICA            — Independent Component Analysis (linear)
  3. NMF            — Non-negative Matrix Factorization (constrained linear)
  4. TruncatedSVD   — Sparse SVD / LSA (linear, works with sparse input)
  5. DiffusionMap   — Coifman & Lafon (2006) — nonlinear manifold
  6. GAHIB           — Proposed (GAT + IB + Lorentz)

All classical methods: scikit-learn implementations.
Latent dim = 10 for all methods to match GAHIB.

Preprocessing: normalize, log1p, 2000 HVGs, subsample 3000 cells (shared).
"""

import sys, os, gc, time, traceback
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import (
    PCA, FastICA, NMF, TruncatedSVD
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.exp_utils import (
    discover_datasets, get_labels, load_and_preprocess,
    get_dense_X, evaluate_latent, get_done_datasets
)
from gahib import GAHIB

# ── Configuration ──
EPOCHS = 200
LATENT_DIM = 10
EXPERIMENT = 'classical_benchmark'
PREFIX = 'classical'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'GAHIB_results', EXPERIMENT)
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

GAHIB_CONFIG = dict(
    recon=1.0, irecon=0.5, lorentz=5.0, beta=0.1,
    encoder_type='graph', graph_type='GAT',
)

METHOD_NAMES = ['PCA', 'ICA', 'NMF', 'TruncatedSVD', 'DiffusionMap', 'GAHIB']


def run_pca(X):
    return PCA(n_components=LATENT_DIM, random_state=42).fit_transform(X)


def run_ica(X):
    return FastICA(n_components=LATENT_DIM, random_state=42,
                   max_iter=500, tol=0.01).fit_transform(X)


def run_nmf(X):
    X_pos = np.clip(X, 0, None)
    return NMF(n_components=LATENT_DIM, random_state=42,
               max_iter=500).fit_transform(X_pos)


def run_truncsvd(X):
    # Works on sparse — but exp_utils already densifies; use sparse for efficiency
    from scipy.sparse import issparse
    return TruncatedSVD(n_components=LATENT_DIM,
                        random_state=42).fit_transform(X)


def run_diffusionmap(X):
    """Diffusion Map via diffusion distance kernel."""
    try:
        from sklearn.neighbors import kneighbors_graph
        from scipy.sparse.linalg import eigs
        from scipy.sparse import diags as sp_diags
        import scipy.sparse as sp

        k = min(15, X.shape[0] - 1)
        A = kneighbors_graph(X, n_neighbors=k, mode='distance',
                             include_self=False)
        A = (A + A.T) / 2  # symmetrize

        # Gaussian kernel on distances
        sigma = np.median(A.data) if len(A.data) > 0 else 1.0
        A.data = np.exp(-(A.data ** 2) / (2 * sigma ** 2))

        # Row-normalize → Markov matrix
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        D_inv = sp_diags(1.0 / row_sums)
        M = D_inv @ A

        # Eigendecomposition
        n_eigs = min(LATENT_DIM + 1, M.shape[0] - 2)
        vals, vecs = eigs(M.astype(np.float64), k=n_eigs, which='LR')
        idx = np.argsort(-np.real(vals))
        vecs = np.real(vecs[:, idx[1:LATENT_DIM + 1]])  # skip trivial eig
        return vecs.astype(np.float32)
    except Exception as e:
        print(f"    DiffusionMap fallback to PCA: {e}")
        return run_pca(X)


def run_classical(name, X):
    """Run a classical method and return latent array."""
    runners = {
        'PCA': run_pca,
        'ICA': run_ica,
        'NMF': run_nmf,
        'TruncatedSVD': run_truncsvd,
        'DiffusionMap': run_diffusionmap,
    }
    try:
        return runners[name](X)
    except Exception as e:
        print(f"    ✗ {name} FAILED: {e}")
        traceback.print_exc()
        return None


def train_gahib(adata1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        m = GAHIB(adata1, layer='counts', hidden_dim=128, latent_dim=LATENT_DIM,
                 i_dim=2, lr=1e-4, loss_type='nb', device=device, **GAHIB_CONFIG)
        m.fit(epochs=EPOCHS, patience=30, early_stop=True, compute_metrics=False)
        latent = m.get_latent()
        res = m.get_resource_metrics()
        del m
        torch.cuda.empty_cache()
        return latent, res['train_time']
    except Exception as e:
        print(f"    ✗ GAHIB FAILED: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None, 0.0


def main():
    datasets = discover_datasets()
    done = get_done_datasets(TABLES_DIR, PREFIX)

    print(f"\n{'='*70}")
    print(f"CLASSICAL DIMENSIONALITY REDUCTION BENCHMARK")
    print(f"Methods: {METHOD_NAMES}")
    print(f"Datasets: {len(datasets)} ({len(done)} already done)")
    print(f"Latent dim: {LATENT_DIM}")
    print(f"{'='*70}\n")

    for filepath in datasets:
        dataset_name = os.path.basename(filepath).replace('.h5ad', '')
        if dataset_name in done:
            print(f"  Skipping {dataset_name} (already done)")
            continue

        print(f"\n{'─'*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─'*60}")

        try:
            adata1 = load_and_preprocess(filepath)
        except Exception as e:
            print(f"  ✗ Preprocess failed: {e}")
            traceback.print_exc()
            continue

        X_dense = get_dense_X(adata1)
        labels, _ = get_labels(adata1)
        all_metrics = []

        # 1-5: Classical methods
        for name in ['PCA', 'ICA', 'NMF', 'TruncatedSVD', 'DiffusionMap']:
            print(f"  Running {name}...")
            t0 = time.time()
            z = run_classical(name, X_dense)
            elapsed = time.time() - t0
            if z is not None:
                mets = evaluate_latent(z, labels)
                mets['train_time'] = elapsed
                print(f"    ✓ {name}: ARI={mets.get('ARI', 0):.3f}, time={elapsed:.1f}s")
            else:
                mets = {}
            all_metrics.append(mets)

        # 6: GAHIB
        print(f"  Training GAHIB...")
        z, elapsed = train_gahib(adata1)
        if z is not None:
            mets = evaluate_latent(z, labels)
            mets['train_time'] = elapsed
            print(f"    ✓ GAHIB: ARI={mets.get('ARI', 0):.3f}, time={elapsed:.1f}s")
        else:
            mets = {}
        all_metrics.append(mets)

        df = pd.DataFrame(all_metrics, index=METHOD_NAMES)
        csv_path = os.path.join(TABLES_DIR, f'{PREFIX}_{dataset_name}_df.csv')
        df.to_csv(csv_path, index_label='method')
        print(f"  Saved: {csv_path}")

        del adata1
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"CLASSICAL BENCHMARK COMPLETE — Results in: {TABLES_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
