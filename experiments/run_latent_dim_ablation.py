#!/usr/bin/env python3
"""
Experiment: Latent Dimension Ablation
=====================================
Sweeps the latent dimension d = [2, 5, 10, 20, 50]
to justify the default choice of d=10.

All other GAHIB hyperparameters held at default.
IB dimension is always 2 (fixed).
Resumable: skips dataset+dim combos already saved.
"""

import sys, os, gc, traceback
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.exp_utils import (
    discover_datasets, get_labels, load_and_preprocess,
    evaluate_latent, get_done_datasets,
)
from gahib import GAHIB

# ── Configuration ──
EPOCHS = 200
EXPERIMENT = 'latent_dim_ablation'
PREFIX = 'latdim'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'GAHIB_results', EXPERIMENT)
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

LATENT_DIMS = [3, 5, 10, 20, 50]

DEFAULTS = dict(
    recon=1.0, irecon=0.5, lorentz=5.0, beta=0.1,
    encoder_type='graph', graph_type='GAT',
    hidden_dim=128, i_dim=2,
    lr=1e-4, loss_type='nb',
)


def run_single(adata1, latent_dim, dataset_name):
    """Train GAHIB with a specific latent dimension."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label = f"d={latent_dim}"
    try:
        model = GAHIB(
            adata1, layer='counts',
            latent_dim=latent_dim,
            device=device,
            **DEFAULTS,
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
        metrics['latent_dim'] = latent_dim

        print(f"    ✓ {label}: NMI={metrics.get('NMI',0):.3f}, "
              f"ARI={metrics.get('ARI',0):.3f}")

        del model
        torch.cuda.empty_cache()
        return metrics

    except Exception as e:
        print(f"    ✗ {label} FAILED: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None


def main():
    datasets = discover_datasets()
    done = get_done_datasets(TABLES_DIR, PREFIX)

    print(f"\n{'='*70}")
    print(f"GAHIB LATENT DIMENSION ABLATION")
    print(f"Dimensions: {LATENT_DIMS}")
    print(f"Datasets: {len(datasets)} ({len(done)} already done)")
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
        for d in LATENT_DIMS:
            metrics = run_single(adata1, d, dataset_name)
            all_metrics.append(metrics if metrics else {'latent_dim': d})

        df = pd.DataFrame(all_metrics)
        csv_path = os.path.join(TABLES_DIR, f'{PREFIX}_{dataset_name}_df.csv')
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        del adata1
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"LATENT DIM ABLATION COMPLETE — Results in: {TABLES_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
