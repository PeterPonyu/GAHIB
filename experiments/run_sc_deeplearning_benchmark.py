#!/usr/bin/env python3
"""
Experiment 8: Single-Cell Deep Learning Benchmark
===================================================
Compares GAHIB against 11 published deep learning baselines from the
external-benchmarker skill, all trained on identical preprocessed data.

Baselines (11 external + 1 proposed):
  1. scVI           — Lopez et al. (2018, Nat Methods) — NB-VAE gold standard
  2. CellBLAST      — Cao et al. (2020, Nat Comms)    — adversarial scRNA alignment
  3. CLEAR          — Han et al. (2022)                — contrastive scRNA learning
  4. SCALEX         — Xiong et al. (2021, Nat Biotech) — online batch integration
  5. scDAC          — Tian et al. (2021)               — deep adaptive clustering
  6. scDeepCluster  — Tian et al. (2019, Nat Methods)  — ZINB deep clustering
  7. scDHMap        — Ding et al. (2023)               — deep hyperbolic manifold
  8. scGNN          — Wang et al. (2021, Nat Comms)    — graph neural scRNA
  9. scGCC          — You et al. (2021)                — graph contrastive clustering
 10. scSMD          — (2022)                           — self-supervised multi-decoder
 11. siVAE          — Kopf et al. (2021)               — mixture-of-experts VAE
 12. GAHIB           — Proposed                         — GAT + IB + Lorentz

Preprocessing: normalize, log1p, 2000 HVGs, subsample 3000 cells (shared).
Each model: 200 epochs x 12 datasets.
"""

import sys, os, gc, time, traceback, types
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.exp_utils import (
    discover_datasets, get_labels, load_and_preprocess,
    get_dense_X, evaluate_latent, get_done_datasets
)
from gahib import GAHIB

# ── External benchmarker import (package alias trick) ──
BENCHMARKER_DIR = os.environ.get(
    "GAHIB_BENCHMARKER_DIR",
    os.path.expanduser("~/.copilot/skills/external-benchmarker"),
)
sys.path.insert(0, os.path.dirname(BENCHMARKER_DIR))
_pkg = types.ModuleType('external_benchmarker')
_pkg.__path__ = [BENCHMARKER_DIR]
_pkg.__package__ = 'external_benchmarker'
sys.modules['external_benchmarker'] = _pkg

try:
    from external_benchmarker.unified_models import (
        create_cellblast_model, create_clear_model, create_scalex_model,
        create_scdac_model, create_scdeepcluster_model, create_scdhmap_model,
        create_scgnn_model, create_scgcc_model, create_scsmd_model, create_sivae_model,
    )
    HAS_BENCHMARKER = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: external-benchmarker not available: {e}")
    HAS_BENCHMARKER = False

try:
    import scvi as scvi_pkg
    HAS_SCVI = True
except ImportError:
    HAS_SCVI = False
    print("Warning: scvi-tools not installed — scVI baseline skipped")

# ── Configuration ──
EPOCHS = 200
BATCH_SIZE = 128
EXPERIMENT = 'sc_deeplearning_benchmark'
PREFIX = 'scdeep'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'GAHIB_results', EXPERIMENT)
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

GAHIB_CONFIG = dict(
    recon=1.0, irecon=0.5, lorentz=5.0, beta=0.1,
    encoder_type='graph', graph_type='GAT',
)

# ── Model method names (logical order: classical → deep → proposed) ──
METHOD_NAMES = [
    'scVI', 'CellBLAST', 'CLEAR', 'SCALEX',
    'scDAC', 'scDeepCluster', 'scDHMap',
    'scGNN', 'scGCC', 'scSMD', 'siVAE',
    'GAHIB',
]


def make_loaders(X_dense, seed=42):
    """Split data into train/val and return DataLoaders.

    Uses TensorDataset(X, X) so every batch is a 2-element tuple
    (x_norm, x_raw).  All unified models handle this format:
    base_model treats element-1 as x_raw, scDHMap/scGNN unpack it, etc.
    """
    n = X_dense.shape[0]
    idx = np.arange(n)
    train_idx, val_idx = train_test_split(idx, test_size=0.15, random_state=seed)
    X_t = torch.FloatTensor(X_dense)
    train_dl = DataLoader(TensorDataset(X_t[train_idx], X_t[train_idx]),
                          batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_t[val_idx], X_t[val_idx]),
                        batch_size=BATCH_SIZE, shuffle=False)
    full_dl = DataLoader(TensorDataset(X_t, X_t),
                         batch_size=BATCH_SIZE, shuffle=False)
    return train_dl, val_dl, full_dl


def train_scvi(adata1, epochs):
    """Train scVI (NB-VAE) and return latent."""
    if not HAS_SCVI:
        return None
    try:
        import anndata as ad
        a = ad.AnnData(X=adata1.layers['counts'].copy())
        scvi_pkg.model.SCVI.setup_anndata(a)
        m = scvi_pkg.model.SCVI(a, n_latent=10, n_hidden=128, n_layers=2)
        m.train(max_epochs=epochs, early_stopping=True,
                early_stopping_patience=25, plan_kwargs={'lr': 1e-3})
        return m.get_latent_representation()
    except Exception as e:
        print(f"    ✗ scVI FAILED: {e}")
        traceback.print_exc()
        return None


# Per-model constructor kwargs (each model has different parameter names)
MODEL_KWARGS = {
    'CellBLAST': dict(hidden_dims=[512, 256]),
    'CLEAR':     dict(hidden_dims=[512, 256]),
    'SCALEX':    dict(hidden_dims=[512, 256]),
    'scDAC':     dict(encoder_dims=[512, 256], decoder_dims=[256, 512]),
    'scDeepCluster': dict(hidden_dims=[512, 256]),
    'scDHMap':   dict(encoder_layers=[512, 256], decoder_layers=[256, 512]),
    'scGNN':     dict(hidden_dim=256),
    'scGCC':     {},   # contrastive — uses own architecture
    'scSMD':     {},   # multi-decoder — uses own architecture
    'siVAE':     dict(hidden_dims=[512, 256]),
}


def train_unified(factory, name, X_dense, epochs):
    """Train one unified external baseline and return latent."""
    if not HAS_BENCHMARKER:
        return None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        train_dl, val_dl, full_dl = make_loaders(X_dense)
        kwargs = MODEL_KWARGS.get(name, {})
        m = factory(input_dim=X_dense.shape[1], latent_dim=10, **kwargs)
        m.fit(train_dl, val_loader=val_dl, epochs=epochs,
              device=str(device), verbose=0, patience=25)
        latent = m.extract_latent(full_dl, device=str(device))['latent']
        del m
        torch.cuda.empty_cache()
        return latent
    except Exception as e:
        print(f"    ✗ {name} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None


def train_gahib(adata1, epochs):
    """Train GAHIB (proposed) and return latent."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        m = GAHIB(adata1, layer='counts', hidden_dim=128, latent_dim=10,
                 i_dim=2, lr=1e-4, loss_type='nb', device=device,
                 **GAHIB_CONFIG)
        m.fit(epochs=epochs, patience=30, early_stop=True, compute_metrics=False)
        latent = m.get_latent()
        del m
        torch.cuda.empty_cache()
        return latent
    except Exception as e:
        print(f"    ✗ GAHIB FAILED: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None


FACTORY_MAP = {}
if HAS_BENCHMARKER:
    FACTORY_MAP = {
        'CellBLAST': create_cellblast_model,
        'CLEAR': create_clear_model,
        'SCALEX': create_scalex_model,
        'scDAC': create_scdac_model,
        'scDeepCluster': create_scdeepcluster_model,
        'scDHMap': create_scdhmap_model,
        'scGNN': create_scgnn_model,
        'scGCC': create_scgcc_model,
        'scSMD': create_scsmd_model,
        'siVAE': create_sivae_model,
    }


def main():
    datasets = discover_datasets()
    done = get_done_datasets(TABLES_DIR, PREFIX)

    print(f"\n{'='*70}")
    print(f"SINGLE-CELL DEEP LEARNING BENCHMARK")
    print(f"Methods: {METHOD_NAMES}")
    print(f"Datasets: {len(datasets)} ({len(done)} already done)")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")
    print(f"scVI available: {HAS_SCVI} | Benchmarker: {HAS_BENCHMARKER}")
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

        # 1. scVI
        print(f"  Training scVI...")
        t0 = time.time()
        z = train_scvi(adata1, EPOCHS)
        elapsed = time.time() - t0
        if z is not None:
            m = evaluate_latent(z, labels)
            m['train_time'] = elapsed
            print(f"    ✓ scVI: ARI={m.get('ARI', 0):.3f}, time={elapsed:.1f}s")
        else:
            m = {}
        all_metrics.append(m)

        # 2-11. Unified external models
        for method_name in ['CellBLAST', 'CLEAR', 'SCALEX', 'scDAC',
                            'scDeepCluster', 'scDHMap', 'scGNN', 'scGCC',
                            'scSMD', 'siVAE']:
            print(f"  Training {method_name}...")
            t0 = time.time()
            factory = FACTORY_MAP.get(method_name)
            z = train_unified(factory, method_name, X_dense, EPOCHS) if factory else None
            elapsed = time.time() - t0
            if z is not None:
                mets = evaluate_latent(z, labels)
                mets['train_time'] = elapsed
                print(f"    ✓ {method_name}: ARI={mets.get('ARI', 0):.3f}, time={elapsed:.1f}s")
            else:
                mets = {}
            all_metrics.append(mets)

        # 12. GAHIB
        print(f"  Training GAHIB...")
        t0 = time.time()
        z = train_gahib(adata1, EPOCHS)
        elapsed = time.time() - t0
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
    print(f"SC DEEP LEARNING BENCHMARK COMPLETE — Results in: {TABLES_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
