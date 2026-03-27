#!/usr/bin/env python3
"""
Experiment 6: Graph Convolution Type Sweep
============================================
Tests different graph convolution operators for the GAHIB encoder, all with
the same GAHIB config (IB + Lorentz + GAT-level hyperparameters).

Variants (6):
  1. GCN          — Kipf & Welling (2017)
  2. GraphSAGE    — Hamilton et al. (2017)
  3. GAT          — Velickovic et al. (2018) [current GAHIB default]
  4. Cheb         — Defferrard et al. (2016), Chebyshev spectral
  5. TAG          — Du et al. (2017), Topology Adaptive
  6. Transformer  — Shi et al. (2021), Graph Transformer

Preprocessing: normalize, log1p, 2000 HVGs, subsample 3000 cells.
Each variant: 200 epochs x 12 datasets.
"""

import sys, os, gc, traceback
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.exp_utils import (
    discover_datasets, get_labels, load_and_preprocess,
    evaluate_latent, get_done_datasets
)
from gahib import GAHIB

# ── Configuration ──
EPOCHS = 200
EXPERIMENT = 'graph_conv_sweep'
PREFIX = 'gconv'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'GAHIB_results', EXPERIMENT)
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

# Shared GAHIB base config (full model, only graph_type varies)
_BASE = dict(
    recon=1.0, irecon=0.5, lorentz=5.0, beta=0.1,
    encoder_type='graph',
)

VARIANTS = {
    'GCN': dict(**_BASE, graph_type='GCN'),
    'GraphSAGE': dict(**_BASE, graph_type='SAGE'),
    'GAT': dict(**_BASE, graph_type='GAT'),
    'Cheb': dict(**_BASE, graph_type='Cheb'),
    'TAG': dict(**_BASE, graph_type='TAG'),
    'GraphTransformer': dict(**_BASE, graph_type='Transformer'),
}


def run_single(adata1, variant_name, params, dataset_name):
    """Train one variant on preprocessed adata1."""
    print(f"  Training {variant_name} on {dataset_name}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = GAHIB(
            adata1, layer='counts',
            hidden_dim=128, latent_dim=10, i_dim=2,
            lr=1e-4, loss_type='nb',
            device=device,
            **params
        )
        model.fit(epochs=EPOCHS, patience=30, early_stop=True,
                  compute_metrics=False)
        latent = model.get_latent()
        labels, _ = get_labels(adata1)
        metrics = evaluate_latent(latent, labels)

        res = model.get_resource_metrics()
        metrics['train_time'] = res['train_time']
        metrics['peak_memory_gb'] = res['peak_memory_gb']
        metrics['actual_epochs'] = res['actual_epochs']

        print(f"    ✓ {variant_name}: ARI={metrics.get('ARI', 0):.3f}, "
              f"NMI={metrics.get('NMI', 0):.3f}, time={res['train_time']:.1f}s")

        del model
        torch.cuda.empty_cache()
        return metrics

    except Exception as e:
        print(f"    ✗ {variant_name} FAILED: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None


def main():
    datasets = discover_datasets()
    done = get_done_datasets(TABLES_DIR, PREFIX)
    method_names = list(VARIANTS.keys())

    print(f"\n{'='*70}")
    print(f"GAHIB GRAPH CONVOLUTION TYPE SWEEP")
    print(f"Variants: {method_names}")
    print(f"Datasets: {len(datasets)} ({len(done)} already done)")
    print(f"Epochs: {EPOCHS}, compute_metrics=False")
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
            print(f"  ✗ Failed to preprocess: {e}")
            traceback.print_exc()
            continue

        all_metrics = []
        for variant_name, params in VARIANTS.items():
            metrics = run_single(adata1, variant_name, params, dataset_name)
            all_metrics.append(metrics if metrics else {})

        df = pd.DataFrame(all_metrics, index=method_names)
        csv_path = os.path.join(TABLES_DIR, f'{PREFIX}_{dataset_name}_df.csv')
        df.to_csv(csv_path, index_label='method')
        print(f"  Saved: {csv_path}")

        del adata1
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"GRAPH CONV SWEEP COMPLETE — Results in: {TABLES_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
