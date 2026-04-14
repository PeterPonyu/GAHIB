#!/usr/bin/env python3
"""Aggregate results from the 4 new experiments into summary CSVs for plotting."""
import os, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def aggregate_sensitivity():
    """Per-sweep, per-value mean metrics across datasets."""
    tables = os.path.join(ROOT, 'GAHIB_results', 'hyperparam_sensitivity', 'tables')
    out_dir = os.path.join(ROOT, 'GAHIB_results', 'hyperparam_sensitivity')

    SWEEPS = {
        'beta':    [0.01, 0.05, 0.1, 0.5, 1.0],
        'lam_ib':  [0.1, 0.25, 0.5, 1.0, 2.0],
        'lam_hyp': [1.0, 2.5, 5.0, 10.0, 20.0],
        'k_nn':    [5, 10, 15, 20, 30],
    }

    for sweep_name, values in SWEEPS.items():
        rows = []
        for val in values:
            pattern = os.path.join(tables, f'hpsens_{sweep_name}_{val}_*_df.csv')
            csvs = glob.glob(pattern)
            if not csvs:
                continue
            dfs = [pd.read_csv(f) for f in csvs]
            combined = pd.concat(dfs, ignore_index=True)
            numeric = combined.select_dtypes(include=[np.number]).mean()
            row = dict(numeric)
            row['sweep_value'] = val
            row['n_datasets'] = len(csvs)
            rows.append(row)
        if rows:
            agg = pd.DataFrame(rows)
            out = os.path.join(out_dir, f'sensitivity_{sweep_name}_summary.csv')
            agg.to_csv(out, index=False)
            print(f'  ✓ {out} ({len(rows)} values, {agg["n_datasets"].iloc[0]} datasets)')


def aggregate_seeds():
    """Per-dataset mean/std across seeds."""
    tables = os.path.join(ROOT, 'GAHIB_results', 'seed_robustness', 'tables')
    out_dir = os.path.join(ROOT, 'GAHIB_results', 'seed_robustness')

    SEEDS = [42, 123, 456, 789, 2024]
    datasets = set()
    for f in glob.glob(os.path.join(tables, 'seed*_*_df.csv')):
        base = os.path.basename(f)
        for s in SEEDS:
            prefix = f'seed{s}_'
            if base.startswith(prefix):
                name = base[len(prefix):].replace('_df.csv', '')
                datasets.add(name)
                break

    rows = []
    for name in sorted(datasets):
        seed_dfs = []
        for s in SEEDS:
            csv = os.path.join(tables, f'seed{s}_{name}_df.csv')
            if os.path.exists(csv):
                seed_dfs.append(pd.read_csv(csv))
        if not seed_dfs:
            continue
        combined = pd.concat(seed_dfs, ignore_index=True)
        numeric = combined.select_dtypes(include=[np.number])
        row = {'dataset': name, 'n_seeds': len(seed_dfs)}
        for col in numeric.columns:
            if col == 'seed':
                continue
            row[f'{col}_mean'] = numeric[col].mean()
            row[f'{col}_std'] = numeric[col].std()
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        out = os.path.join(out_dir, 'seed_robustness_summary.csv')
        df.to_csv(out, index=False)
        print(f'  ✓ {out} ({len(rows)} datasets, 5 seeds each)')


def aggregate_cost():
    """Per-method cost stats."""
    tables = os.path.join(ROOT, 'GAHIB_results', 'computational_cost', 'tables')
    out_dir = os.path.join(ROOT, 'GAHIB_results', 'computational_cost')

    csvs = glob.glob(os.path.join(tables, 'compcost_*_df.csv'))
    if csvs:
        all_df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
        summary = all_df.groupby('method').agg({
            'train_time': ['mean', 'std'],
            'peak_memory_gb': ['mean', 'std'],
            'actual_epochs': ['mean', 'std'],
        }).round(3)
        out = os.path.join(out_dir, 'cost_summary.csv')
        summary.to_csv(out)
        print(f'  ✓ {out} ({len(csvs)} datasets, {len(summary)} methods)')

    scaling_csvs = glob.glob(os.path.join(
        ROOT, 'GAHIB_results', 'computational_cost', 'scaling', 'scaling_*.csv'))
    if scaling_csvs:
        all_scale = pd.concat([pd.read_csv(f) for f in scaling_csvs], ignore_index=True)
        out = os.path.join(out_dir, 'scaling_summary.csv')
        all_scale.to_csv(out, index=False)
        print(f'  ✓ {out} ({len(scaling_csvs)} datasets × 4 sizes)')


if __name__ == '__main__':
    print('Aggregating HP sensitivity...')
    aggregate_sensitivity()
    print('Aggregating seed robustness...')
    aggregate_seeds()
    print('Aggregating computational cost...')
    aggregate_cost()
    print('Done.')
