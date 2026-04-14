#!/usr/bin/env python3
"""
Experiment: Batch-Size Cost Analysis
====================================
Sweep GAHIB training cost across batch sizes on a fixed-size subsample
of a representative set of datasets.

Unlike run_computational_cost.py (which varies cell count at the
default batch size), this script fixes cell count and varies the
mini-batch size used by the GAT encoder.  The per-batch wall-clock and
peak GPU memory produced here answer: "how does the optimiser step
cost scale with batch size?"

Resumable: per-dataset CSV saved individually.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.exp_utils import discover_datasets, load_and_preprocess
from gahib import GAHIB

EXPERIMENT = "computational_cost"
PREFIX = "bsens"
EPOCHS = 100
N_CELLS = 2000
BATCH_SIZES = [32, 64, 128, 256, 512]
N_DATASETS = 5

RESULTS_DIR = os.path.join(PROJECT_ROOT, "GAHIB_results", EXPERIMENT)
BATCH_DIR = os.path.join(RESULTS_DIR, "batch_size")
os.makedirs(BATCH_DIR, exist_ok=True)


def run_batch_single(adata, batch_size, dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        t0 = time.time()
        model = GAHIB(
            adata, layer="counts",
            recon=1.0, irecon=0.5, lorentz=5.0, beta=0.1,
            encoder_type="graph", graph_type="GAT",
            hidden_dim=128, latent_dim=10, i_dim=2,
            batch_size=batch_size,
            lr=1e-4, loss_type="nb",
            device=device,
        )
        model.fit(epochs=EPOCHS, patience=30, early_stop=True,
                  compute_metrics=False)
        wall = time.time() - t0
        res = model.get_resource_metrics()

        result = {
            "dataset": dataset_name,
            "batch_size": batch_size,
            "n_cells": adata.n_obs,
            "train_time_s": res["train_time"],
            "wall_time_s": wall,
            "peak_memory_gb": res["peak_memory_gb"],
            "actual_epochs": res["actual_epochs"],
            "time_per_epoch_s": res["train_time"] / max(1, res["actual_epochs"]),
        }
        print(f"    bs={batch_size:3d}: "
              f"{res['train_time']:6.1f}s total, "
              f"{result['time_per_epoch_s']:5.2f}s/epoch, "
              f"{res['peak_memory_gb']:.3f}GB")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    except Exception as e:
        print(f"    bs={batch_size} FAILED: {e}")
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"dataset": dataset_name, "batch_size": batch_size,
                "n_cells": adata.n_obs if adata is not None else -1}


def main():
    datasets = discover_datasets()[:N_DATASETS]
    print(f"\n{'='*70}")
    print(f"GAHIB BATCH-SIZE COST SWEEP")
    print(f"Fixed N_CELLS={N_CELLS}, batch sizes={BATCH_SIZES}")
    print(f"Datasets: {len(datasets)}")
    print(f"{'='*70}\n")

    rng = np.random.RandomState(42)
    per_ds_results = []

    for filepath in datasets:
        dataset_name = os.path.basename(filepath).replace(".h5ad", "")
        csv_path = os.path.join(BATCH_DIR, f"{PREFIX}_{dataset_name}.csv")
        if os.path.exists(csv_path):
            print(f"  skipping {dataset_name} (already done)")
            per_ds_results.append(pd.read_csv(csv_path))
            continue

        print(f"\n─── {dataset_name} ───")
        try:
            adata = load_and_preprocess(filepath)
        except Exception as e:
            print(f"  preprocess failed: {e}")
            continue

        # Subsample to fixed cell count so cost differences come from
        # batch size alone, not dataset size.
        if adata.n_obs > N_CELLS:
            idx = rng.choice(adata.n_obs, N_CELLS, replace=False)
            adata_sub = adata[idx].copy()
        else:
            adata_sub = adata.copy()

        ds_rows = []
        for bs in BATCH_SIZES:
            if bs > adata_sub.n_obs:
                continue
            ds_rows.append(run_batch_single(adata_sub, bs, dataset_name))

        ds_df = pd.DataFrame(ds_rows)
        ds_df.to_csv(csv_path, index=False)
        per_ds_results.append(ds_df)
        print(f"  saved {csv_path}")

        del adata, adata_sub
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if per_ds_results:
        full = pd.concat(per_ds_results, ignore_index=True)
        summary = (full.groupby("batch_size")
                       .agg(train_time_mean=("train_time_s", "mean"),
                            train_time_std=("train_time_s", "std"),
                            time_per_epoch_mean=("time_per_epoch_s", "mean"),
                            time_per_epoch_std=("time_per_epoch_s", "std"),
                            peak_mem_mean=("peak_memory_gb", "mean"),
                            peak_mem_std=("peak_memory_gb", "std"),
                            n=("train_time_s", "count"))
                       .reset_index())
        summary.to_csv(os.path.join(RESULTS_DIR, "batch_size_summary.csv"),
                       index=False)
        full.to_csv(os.path.join(RESULTS_DIR, "batch_size_full.csv"),
                    index=False)
        print(f"\nBATCH-SIZE SWEEP SUMMARY")
        print(summary.to_string(index=False))

    print(f"\n{'='*70}")
    print(f"BATCH-SIZE COST SWEEP COMPLETE — {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
