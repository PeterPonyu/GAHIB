#!/usr/bin/env python3
"""
Experiment: Hyperparameter Sensitivity Analysis
================================================
Sweeps key hyperparameters of GAHIB to demonstrate robustness:
  - β (KL weight):        [0.01, 0.05, 0.1, 0.5, 1.0]
  - λ_ib (IB weight):     [0.1, 0.25, 0.5, 1.0, 2.0]
  - λ_hyp (Lorentz wt):   [1.0, 2.5, 5.0, 10.0, 20.0]
  - k (kNN neighbours):   [5, 10, 15, 20, 30]

Each sweep varies ONE parameter while holding the others at default.
Resumable: skips datasets+sweep combos that already have saved CSVs.
"""

import sys, os, gc, traceback, time, json
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
EXPERIMENT = 'hyperparam_sensitivity'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'GAHIB_results', EXPERIMENT)
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

# Default GAHIB configuration
DEFAULTS = dict(
    recon=1.0, irecon=0.5, lorentz=5.0, beta=0.1,
    encoder_type='graph', graph_type='GAT',
    hidden_dim=128, latent_dim=10, i_dim=2,
    lr=1e-4, loss_type='nb',
)

# Sweep definitions: param_name -> (GAHIB kwarg, values)
SWEEPS = {
    'beta':   ('beta',    [0.01, 0.05, 0.1, 0.5, 1.0]),
    'lam_ib': ('irecon',  [0.1, 0.25, 0.5, 1.0, 2.0]),
    'lam_hyp':('lorentz', [1.0, 2.5, 5.0, 10.0, 20.0]),
    'k_nn':   ('k_nn',    [5, 10, 15, 20, 30]),
}


def _make_prefix(sweep_name, val):
    return f"hpsens_{sweep_name}_{val}"


def _is_done(sweep_name, val, dataset_name):
    prefix = _make_prefix(sweep_name, val)
    csv = os.path.join(TABLES_DIR, f'{prefix}_{dataset_name}_df.csv')
    return os.path.exists(csv)


def run_single(adata1, params, dataset_name, label):
    """Train one GAHIB config and return metrics dict."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Separate k_nn from model params (handled via GAHIB's n_neighbors)
        model_params = {k: v for k, v in params.items() if k != 'k_nn'}
        k_nn = params.get('k_nn', 15)

        model = GAHIB(
            adata1, layer='counts',
            device=device,
            n_neighbors=k_nn,
            **model_params,
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

        print(f"    ✓ {label}: NMI={metrics.get('NMI',0):.3f}, "
              f"ARI={metrics.get('ARI',0):.3f}, time={res['train_time']:.1f}s")

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

    print(f"\n{'='*70}")
    print(f"GAHIB HYPERPARAMETER SENSITIVITY ANALYSIS")
    print(f"Sweeps: {list(SWEEPS.keys())}")
    print(f"Datasets: {len(datasets)}")
    print(f"{'='*70}\n")

    for filepath in datasets:
        dataset_name = os.path.basename(filepath).replace('.h5ad', '')

        # Check if ALL sweeps for this dataset are done
        all_done = all(
            _is_done(sweep_name, val, dataset_name)
            for sweep_name, (_, values) in SWEEPS.items()
            for val in values
        )
        if all_done:
            print(f"  Skipping {dataset_name} (all sweeps done)")
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

        for sweep_name, (param_key, values) in SWEEPS.items():
            for val in values:
                if _is_done(sweep_name, val, dataset_name):
                    continue

                label = f"{sweep_name}={val}"
                params = dict(DEFAULTS)
                params[param_key] = val

                metrics = run_single(adata1, params, dataset_name, label)
                if metrics is None:
                    metrics = {}
                metrics['sweep_param'] = sweep_name
                metrics['sweep_value'] = val

                # Save individual result
                prefix = _make_prefix(sweep_name, val)
                df = pd.DataFrame([metrics])
                csv_path = os.path.join(
                    TABLES_DIR, f'{prefix}_{dataset_name}_df.csv')
                df.to_csv(csv_path, index=False)

        del adata1
        gc.collect()
        torch.cuda.empty_cache()

    # ── Aggregate results ──
    print(f"\n{'='*70}")
    print("Aggregating sweep results...")
    for sweep_name, (_, values) in SWEEPS.items():
        rows = []
        for val in values:
            prefix = _make_prefix(sweep_name, val)
            csvs = [f for f in os.listdir(TABLES_DIR)
                    if f.startswith(prefix) and f.endswith('_df.csv')]
            if not csvs:
                continue
            dfs = [pd.read_csv(os.path.join(TABLES_DIR, f)) for f in csvs]
            combined = pd.concat(dfs, ignore_index=True)
            summary = combined.select_dtypes(include=[np.number]).mean()
            summary['sweep_value'] = val
            summary['n_datasets'] = len(csvs)
            rows.append(summary)
        if rows:
            agg = pd.DataFrame(rows)
            agg_path = os.path.join(RESULTS_DIR,
                                    f'sensitivity_{sweep_name}_summary.csv')
            agg.to_csv(agg_path, index=False)
            print(f"  {sweep_name}: {agg_path}")

    print(f"\nHYPERPARAMETER SENSITIVITY COMPLETE — {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
