#!/usr/bin/env python3
"""
Experiment: Multi-Seed Robustness Analysis
==========================================
Trains GAHIB with 5 different random seeds to demonstrate
reproducibility and report mean±std across seeds.

Seeds: [42, 123, 456, 789, 2024]
Resumable: per-seed results saved individually.
"""

import sys, os, gc, traceback
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.exp_utils import (
    discover_datasets, get_labels, load_and_preprocess,
    evaluate_latent,
)
from gahib import GAHIB

# ── Configuration ──
EPOCHS = 200
EXPERIMENT = 'seed_robustness'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'GAHIB_results', EXPERIMENT)
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]

DEFAULTS = dict(
    recon=1.0, irecon=0.5, lorentz=5.0, beta=0.1,
    encoder_type='graph', graph_type='GAT',
    hidden_dim=128, latent_dim=10, i_dim=2,
    lr=1e-4, loss_type='nb',
)


def _seed_prefix(seed):
    return f"seed{seed}"


def _is_seed_done(seed, dataset_name):
    prefix = _seed_prefix(seed)
    csv = os.path.join(TABLES_DIR, f'{prefix}_{dataset_name}_df.csv')
    return os.path.exists(csv)


def run_single(adata1, seed, dataset_name):
    """Train GAHIB with a specific random seed."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = GAHIB(
            adata1, layer='counts',
            random_seed=seed,
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
        metrics['seed'] = seed

        print(f"    ✓ seed={seed}: NMI={metrics.get('NMI',0):.3f}, "
              f"ARI={metrics.get('ARI',0):.3f}")

        del model
        torch.cuda.empty_cache()
        return metrics

    except Exception as e:
        print(f"    ✗ seed={seed} FAILED: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None


def main():
    datasets = discover_datasets()

    print(f"\n{'='*70}")
    print(f"GAHIB MULTI-SEED ROBUSTNESS ANALYSIS")
    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {len(datasets)}")
    print(f"{'='*70}\n")

    for filepath in datasets:
        dataset_name = os.path.basename(filepath).replace('.h5ad', '')

        all_done = all(_is_seed_done(s, dataset_name) for s in SEEDS)
        if all_done:
            print(f"  Skipping {dataset_name} (all seeds done)")
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

        for seed in SEEDS:
            if _is_seed_done(seed, dataset_name):
                print(f"    Skipping seed={seed} (done)")
                continue

            metrics = run_single(adata1, seed, dataset_name)
            if metrics is None:
                metrics = {'seed': seed}

            prefix = _seed_prefix(seed)
            df = pd.DataFrame([metrics])
            csv_path = os.path.join(
                TABLES_DIR, f'{prefix}_{dataset_name}_df.csv')
            df.to_csv(csv_path, index=False)

        del adata1
        gc.collect()
        torch.cuda.empty_cache()

    # ── Aggregate across seeds ──
    print(f"\n{'='*70}")
    print("Aggregating seed results...")
    agg_rows = []
    for filepath in datasets:
        dataset_name = os.path.basename(filepath).replace('.h5ad', '')
        seed_dfs = []
        for seed in SEEDS:
            csv = os.path.join(
                TABLES_DIR, f'{_seed_prefix(seed)}_{dataset_name}_df.csv')
            if os.path.exists(csv):
                seed_dfs.append(pd.read_csv(csv))
        if not seed_dfs:
            continue
        combined = pd.concat(seed_dfs, ignore_index=True)
        numeric = combined.select_dtypes(include=[np.number])
        row = {}
        row['dataset'] = dataset_name
        row['n_seeds'] = len(seed_dfs)
        for col in numeric.columns:
            if col == 'seed':
                continue
            row[f'{col}_mean'] = numeric[col].mean()
            row[f'{col}_std'] = numeric[col].std()
        agg_rows.append(row)

    if agg_rows:
        agg_df = pd.DataFrame(agg_rows)
        agg_path = os.path.join(RESULTS_DIR, 'seed_robustness_summary.csv')
        agg_df.to_csv(agg_path, index=False)
        print(f"  Summary: {agg_path}")

        # Print key stats
        key_metrics = ['NMI', 'ARI', 'ASW']
        for m in key_metrics:
            mean_col = f'{m}_mean'
            std_col = f'{m}_std'
            if mean_col in agg_df.columns:
                avg_mean = agg_df[mean_col].mean()
                avg_std = agg_df[std_col].mean()
                print(f"  {m}: {avg_mean:.3f} ± {avg_std:.4f} (avg across datasets)")

    print(f"\nSEED ROBUSTNESS COMPLETE — {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
