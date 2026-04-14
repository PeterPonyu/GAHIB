#!/usr/bin/env python3
"""
Sequential runner for all 4 new GAHIB experiments.
===================================================
Runs experiments ONE AT A TIME to fully utilise the GPU.
Skips expensive DRE/LSE metrics — only computes fast clustering metrics
(NMI, ARI, ASW, DAV, CAL, COR) to save CPU resources.

Full metrics can be computed post-hoc on saved latents.

Fully resumable: each atomic result is saved to its own CSV.
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


# ── Shared training function ──
EPOCHS = 200
GAHIB_DEFAULTS = dict(
    recon=1.0, irecon=0.5, lorentz=5.0, beta=0.1,
    encoder_type='graph', graph_type='GAT',
    hidden_dim=128, latent_dim=10, i_dim=2,
    lr=1e-4, loss_type='nb',
)


def train_and_eval(adata1, params, label=''):
    """Train GAHIB with params, return fast metrics + resource info."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Separate special keys from model constructor params
        model_params = {k: v for k, v in params.items()
                        if k not in ('k_nn',)}
        n_neighbors = params.get('k_nn', 15)

        model = GAHIB(
            adata1, layer='counts', device=device,
            n_neighbors=n_neighbors, **model_params,
        )
        model.fit(epochs=EPOCHS, patience=20, val_every=1, early_stop=True,
                  compute_metrics=False)

        latent = model.get_latent()
        labels, _ = get_labels(adata1)
        metrics = evaluate_latent(latent, labels)

        res = model.get_resource_metrics()
        metrics['train_time'] = res['train_time']
        metrics['peak_memory_gb'] = res['peak_memory_gb']
        metrics['actual_epochs'] = res['actual_epochs']

        print(f"    ✓ {label}: NMI={metrics['NMI']:.3f}, "
              f"ASW={metrics['ASW']:.3f}, {res['train_time']:.1f}s")

        del model
        torch.cuda.empty_cache()
        return metrics
    except Exception as e:
        print(f"    ✗ {label}: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None


# =====================================================================
# Experiment 1: Latent Dimension Ablation
# =====================================================================
def run_latent_dim():
    DIMS = [3, 5, 10, 20, 50]
    tables = os.path.join(PROJECT_ROOT, 'GAHIB_results', 'latent_dim_ablation', 'tables')
    os.makedirs(tables, exist_ok=True)
    done = get_done_datasets(tables, 'latdim')

    print(f"\n{'='*60}\n  EXP 1: LATENT DIMENSION ABLATION  d={DIMS}\n{'='*60}")

    for fp in discover_datasets():
        name = os.path.basename(fp).replace('.h5ad', '')
        if name in done:
            print(f"  Skip {name}")
            continue
        try:
            adata1 = load_and_preprocess(fp)
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            continue

        rows = []
        for d in DIMS:
            p = dict(GAHIB_DEFAULTS)
            p['latent_dim'] = d
            m = train_and_eval(adata1, p, f'd={d}')
            row = m if m else {}
            row['latent_dim'] = d
            rows.append(row)

        pd.DataFrame(rows).to_csv(
            os.path.join(tables, f'latdim_{name}_df.csv'), index=False)
        del adata1; gc.collect(); torch.cuda.empty_cache()


# =====================================================================
# Experiment 2: Seed Robustness
# =====================================================================
def run_seeds():
    SEEDS = [42, 123, 456, 789, 2024]
    tables = os.path.join(PROJECT_ROOT, 'GAHIB_results', 'seed_robustness', 'tables')
    os.makedirs(tables, exist_ok=True)

    print(f"\n{'='*60}\n  EXP 2: SEED ROBUSTNESS  seeds={SEEDS}\n{'='*60}")

    for fp in discover_datasets():
        name = os.path.basename(fp).replace('.h5ad', '')
        all_done = all(os.path.exists(
            os.path.join(tables, f'seed{s}_{name}_df.csv')) for s in SEEDS)
        if all_done:
            print(f"  Skip {name}")
            continue
        try:
            adata1 = load_and_preprocess(fp)
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            continue

        for s in SEEDS:
            csv = os.path.join(tables, f'seed{s}_{name}_df.csv')
            if os.path.exists(csv):
                continue
            p = dict(GAHIB_DEFAULTS)
            p['random_seed'] = s
            m = train_and_eval(adata1, p, f'seed={s}')
            if m is None:
                m = {}
            m['seed'] = s
            pd.DataFrame([m]).to_csv(csv, index=False)

        del adata1; gc.collect(); torch.cuda.empty_cache()


# =====================================================================
# Experiment 3: Computational Cost
# =====================================================================
def run_cost():
    METHODS = {
        'GAHIB': dict(recon=1.0, irecon=0.5, lorentz=5.0, beta=0.1,
                       encoder_type='graph', graph_type='GAT'),
        'Base VAE': dict(recon=1.0, irecon=0.0, lorentz=0.0, beta=1.0,
                          encoder_type='mlp'),
        'VAE+IB+Hyp': dict(recon=1.0, irecon=1.0, lorentz=5.0, beta=1.0,
                             encoder_type='mlp'),
    }
    tables = os.path.join(PROJECT_ROOT, 'GAHIB_results', 'computational_cost', 'tables')
    os.makedirs(tables, exist_ok=True)
    done = get_done_datasets(tables, 'compcost')

    print(f"\n{'='*60}\n  EXP 3: COMPUTATIONAL COST\n{'='*60}")

    for fp in discover_datasets():
        name = os.path.basename(fp).replace('.h5ad', '')
        if name in done:
            print(f"  Skip {name}")
            continue
        try:
            adata1 = load_and_preprocess(fp)
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            continue

        rows = []
        for mname, params in METHODS.items():
            p = dict(GAHIB_DEFAULTS)
            p.update(params)
            m = train_and_eval(adata1, p, mname)
            row = m if m else {}
            row['method'] = mname
            row['n_cells'] = adata1.n_obs
            rows.append(row)

        pd.DataFrame(rows).to_csv(
            os.path.join(tables, f'compcost_{name}_df.csv'), index=False)
        del adata1; gc.collect(); torch.cuda.empty_cache()

    # Scaling analysis on first 10 datasets
    scaling_dir = os.path.join(PROJECT_ROOT, 'GAHIB_results', 'computational_cost', 'scaling')
    os.makedirs(scaling_dir, exist_ok=True)
    SIZES = [500, 1000, 2000, 3000]

    print(f"\n  Scaling analysis: sizes={SIZES}")
    for fp in discover_datasets()[:10]:
        name = os.path.basename(fp).replace('.h5ad', '')
        csv = os.path.join(scaling_dir, f'scaling_{name}.csv')
        if os.path.exists(csv):
            print(f"  Skip scaling {name}")
            continue
        try:
            adata1 = load_and_preprocess(fp)
        except Exception:
            continue

        rows = []
        for n in SIZES:
            if adata1.n_obs > n:
                np.random.seed(42)
                idx = np.random.choice(adata1.n_obs, n, replace=False)
                asub = adata1[idx].copy()
            else:
                asub = adata1.copy()

            m = train_and_eval(asub, dict(GAHIB_DEFAULTS), f'n={n}')
            row = m if m else {}
            row['target_cells'] = n
            row['actual_cells'] = asub.n_obs
            rows.append(row)
            del asub

        pd.DataFrame(rows).to_csv(csv, index=False)
        del adata1; gc.collect(); torch.cuda.empty_cache()


# =====================================================================
# Experiment 4: Hyperparameter Sensitivity
# =====================================================================
def run_sensitivity():
    SWEEPS = {
        'beta':    ('beta',    [0.01, 0.05, 0.1, 0.5, 1.0]),
        'lam_ib':  ('irecon',  [0.1, 0.25, 0.5, 1.0, 2.0]),
        'lam_hyp': ('lorentz', [1.0, 2.5, 5.0, 10.0, 20.0]),
        'k_nn':    ('k_nn',    [5, 10, 15, 20, 30]),
    }
    tables = os.path.join(PROJECT_ROOT, 'GAHIB_results', 'hyperparam_sensitivity', 'tables')
    os.makedirs(tables, exist_ok=True)

    print(f"\n{'='*60}\n  EXP 4: HYPERPARAMETER SENSITIVITY\n{'='*60}")

    for fp in discover_datasets():
        name = os.path.basename(fp).replace('.h5ad', '')

        all_done = all(
            os.path.exists(os.path.join(tables, f'hpsens_{sn}_{v}_{name}_df.csv'))
            for sn, (_, vals) in SWEEPS.items() for v in vals
        )
        if all_done:
            print(f"  Skip {name}")
            continue

        try:
            adata1 = load_and_preprocess(fp)
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            continue

        for sn, (pk, vals) in SWEEPS.items():
            for v in vals:
                csv = os.path.join(tables, f'hpsens_{sn}_{v}_{name}_df.csv')
                if os.path.exists(csv):
                    continue
                p = dict(GAHIB_DEFAULTS)
                p[pk] = v
                m = train_and_eval(adata1, p, f'{sn}={v}')
                row = m if m else {}
                row['sweep_param'] = sn
                row['sweep_value'] = v
                pd.DataFrame([row]).to_csv(csv, index=False)

        del adata1; gc.collect(); torch.cuda.empty_cache()


# =====================================================================
# Main: run all 4 sequentially
# =====================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default=None,
                        help='Run only group A (latdim+cost) or B (seeds+hp)')
    args = parser.parse_args()

    t0 = time.time()
    group = args.group.upper() if args.group else None

    print(f"\n{'#'*60}")
    print(f"  GAHIB NEW EXPERIMENTS — GPU Runner")
    print(f"  Group: {group or 'ALL'}")
    print(f"  Fast clustering metrics only (no DRE/LSE)")
    print(f"  Fully resumable — skips existing results")
    print(f"{'#'*60}\n")

    if group is None or group == 'A':
        run_latent_dim()
        run_cost()
    if group is None or group == 'B':
        run_seeds()
    if group is None or group == 'C':
        run_sensitivity()
    if group == 'BC':
        run_seeds()
        run_sensitivity()

    elapsed = time.time() - t0
    hours = elapsed / 3600
    print(f"\n{'#'*60}")
    print(f"  GROUP {group or 'ALL'} COMPLETE in {hours:.1f} hours")
    print(f"{'#'*60}")
