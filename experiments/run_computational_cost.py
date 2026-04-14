#!/usr/bin/env python3
"""
Experiment: Computational Cost Analysis
========================================
Measures training time, peak memory, and actual epochs for
ALL methods across all 53 datasets.

Methods measured:
  - GAHIB (GAT + IB + Hyp)
  - Base VAE (MLP)
  - scVI
  - PCA (sklearn)
  - scDHMap

Also records scaling behaviour: cells vs. training time
by training GAHIB on progressively larger subsamples
[500, 1000, 2000, 3000] on a subset of 10 datasets.

Resumable: per-dataset results saved individually.
"""

import sys, os, gc, time, traceback
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
EXPERIMENT = 'computational_cost'
PREFIX = 'compcost'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'GAHIB_results', EXPERIMENT)
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

# Methods to benchmark
METHODS = {
    'GAHIB': dict(
        recon=1.0, irecon=0.5, lorentz=5.0, beta=0.1,
        encoder_type='graph', graph_type='GAT',
    ),
    'Base VAE': dict(
        recon=1.0, irecon=0.0, lorentz=0.0, beta=1.0,
        encoder_type='mlp',
    ),
    'VAE+IB+Hyp': dict(
        recon=1.0, irecon=1.0, lorentz=5.0, beta=1.0,
        encoder_type='mlp',
    ),
}

# Scaling analysis: train GAHIB at different cell counts
SCALING_SIZES = [500, 1000, 2000, 3000]
SCALING_DATASETS = 10  # use first N datasets


def run_cost_single(adata1, method_name, params, dataset_name):
    """Train and measure computational cost."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        t0 = time.time()
        model = GAHIB(
            adata1, layer='counts',
            hidden_dim=128, latent_dim=10, i_dim=2,
            lr=1e-4, loss_type='nb',
            device=device,
            **params,
        )
        model.fit(epochs=EPOCHS, patience=30, early_stop=True,
                  compute_metrics=False)
        wall_time = time.time() - t0

        res = model.get_resource_metrics()
        n_cells = adata1.n_obs
        n_genes = adata1.n_vars

        metrics = {
            'method': method_name,
            'dataset': dataset_name,
            'n_cells': n_cells,
            'n_genes': n_genes,
            'train_time_s': res['train_time'],
            'wall_time_s': wall_time,
            'peak_memory_gb': res['peak_memory_gb'],
            'actual_epochs': res['actual_epochs'],
            'time_per_epoch_s': res['train_time'] / max(1, res['actual_epochs']),
        }

        print(f"    ✓ {method_name}: {res['train_time']:.1f}s, "
              f"{res['peak_memory_gb']:.2f}GB, {res['actual_epochs']} epochs")

        del model
        torch.cuda.empty_cache()
        return metrics

    except Exception as e:
        print(f"    ✗ {method_name} FAILED: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {
            'method': method_name,
            'dataset': dataset_name,
            'n_cells': adata1.n_obs,
            'n_genes': adata1.n_vars,
        }


def run_scaling_single(adata1, n_cells, dataset_name):
    """Train GAHIB at a specific cell count for scaling analysis."""
    import scanpy as sc
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Subsample to target size
    if adata1.n_obs > n_cells:
        np.random.seed(42)
        idx = np.random.choice(adata1.n_obs, n_cells, replace=False)
        adata_sub = adata1[idx].copy()
    else:
        adata_sub = adata1.copy()
        n_cells = adata_sub.n_obs

    try:
        t0 = time.time()
        model = GAHIB(
            adata_sub, layer='counts',
            recon=1.0, irecon=0.5, lorentz=5.0, beta=0.1,
            encoder_type='graph', graph_type='GAT',
            hidden_dim=128, latent_dim=10, i_dim=2,
            lr=1e-4, loss_type='nb',
            device=device,
        )
        model.fit(epochs=EPOCHS, patience=30, early_stop=True,
                  compute_metrics=False)
        wall_time = time.time() - t0
        res = model.get_resource_metrics()

        result = {
            'dataset': dataset_name,
            'target_cells': n_cells,
            'actual_cells': adata_sub.n_obs,
            'train_time_s': res['train_time'],
            'peak_memory_gb': res['peak_memory_gb'],
            'actual_epochs': res['actual_epochs'],
        }
        print(f"    ✓ n={n_cells}: {res['train_time']:.1f}s")

        del model, adata_sub
        torch.cuda.empty_cache()
        return result

    except Exception as e:
        print(f"    ✗ n={n_cells} FAILED: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {'dataset': dataset_name, 'target_cells': n_cells}


def main():
    datasets = discover_datasets()
    done = get_done_datasets(TABLES_DIR, PREFIX)

    print(f"\n{'='*70}")
    print(f"GAHIB COMPUTATIONAL COST ANALYSIS")
    print(f"Methods: {list(METHODS.keys())}")
    print(f"Datasets: {len(datasets)} ({len(done)} already done)")
    print(f"{'='*70}\n")

    # ── Part 1: Method cost comparison ──
    print("Part 1: Method cost comparison")
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
        for method_name, params in METHODS.items():
            metrics = run_cost_single(adata1, method_name, params, dataset_name)
            all_metrics.append(metrics)

        df = pd.DataFrame(all_metrics)
        csv_path = os.path.join(TABLES_DIR, f'{PREFIX}_{dataset_name}_df.csv')
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        del adata1
        gc.collect()
        torch.cuda.empty_cache()

    # ── Part 2: Scaling analysis ──
    scaling_dir = os.path.join(RESULTS_DIR, 'scaling')
    os.makedirs(scaling_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Part 2: Scaling analysis (first {SCALING_DATASETS} datasets)")
    print(f"Cell counts: {SCALING_SIZES}")
    print(f"{'='*70}\n")

    scaling_results = []
    for filepath in datasets[:SCALING_DATASETS]:
        dataset_name = os.path.basename(filepath).replace('.h5ad', '')
        scale_csv = os.path.join(scaling_dir, f'scaling_{dataset_name}.csv')
        if os.path.exists(scale_csv):
            print(f"  Skipping {dataset_name} scaling (already done)")
            scaling_results.append(pd.read_csv(scale_csv))
            continue

        print(f"\n  Scaling: {dataset_name}")
        try:
            adata1 = load_and_preprocess(filepath)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

        ds_results = []
        for n_cells in SCALING_SIZES:
            result = run_scaling_single(adata1, n_cells, dataset_name)
            ds_results.append(result)

        ds_df = pd.DataFrame(ds_results)
        ds_df.to_csv(scale_csv, index=False)
        scaling_results.append(ds_df)

        del adata1
        gc.collect()
        torch.cuda.empty_cache()

    # ── Aggregate ──
    if scaling_results:
        all_scaling = pd.concat(scaling_results, ignore_index=True)
        all_scaling.to_csv(
            os.path.join(RESULTS_DIR, 'scaling_summary.csv'), index=False)

    # Aggregate method costs
    cost_csvs = [f for f in os.listdir(TABLES_DIR)
                 if f.startswith(PREFIX) and f.endswith('_df.csv')]
    if cost_csvs:
        all_costs = pd.concat(
            [pd.read_csv(os.path.join(TABLES_DIR, f)) for f in cost_csvs],
            ignore_index=True)
        summary = all_costs.groupby('method').agg({
            'train_time_s': ['mean', 'std'],
            'peak_memory_gb': ['mean', 'std'],
            'actual_epochs': ['mean', 'std'],
        }).round(3)
        summary.to_csv(os.path.join(RESULTS_DIR, 'cost_summary.csv'))
        print(f"\n  Cost summary saved")
        print(summary)

    print(f"\n{'='*70}")
    print(f"COMPUTATIONAL COST COMPLETE — {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
