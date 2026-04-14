#!/usr/bin/env python3
"""
Pseudotime / trajectory validation for GAHIB's Lorentz hierarchy.
=================================================================

Procedure (following scDHMap/scanpy conventions):
  1. Train GAHIB on a dataset with a clear differentiation trajectory
     (dentate gyrus, hematopoiesis).
  2. Extract the Lorentz norm r_i per cell as GAHIB's pseudotime.
  3. Compute Scanpy's diffusion pseudotime (DPT) as the reference.
  4. Compute competing pseudotimes:
       - PCA-based (distance from first PC)
       - UMAP-based (Euclidean distance from root)
       - scDHMap-style hyperbolic norm from HeLIB (if available; else skip)
  5. Evaluate each pseudotime against DPT via:
       - Spearman rank correlation
       - Kendall tau
       - Normalised mean-absolute-rank distance
  6. Save per-dataset results + aggregated summary.

Output: GAHIB_results/trajectory/tables/traj_{dataset}_df.csv
        GAHIB_results/trajectory/trajectory_summary.csv
"""

import os, sys, logging, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import scanpy as sc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp_utils import (
    discover_datasets, load_and_preprocess, get_labels,
)
from gahib import GAHIB

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──
EPOCHS = 200
RESULTS_DIR = PROJECT_ROOT / "GAHIB_results" / "trajectory"
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Datasets with clear developmental trajectories
# (name match key, organism, description)
TARGET_DATASETS = [
    ("dentate", "mouse", "Dentate gyrus — granule cell neurogenesis"),
    ("hemato", "mouse", "Hematopoiesis — HSC → differentiated lineages"),
    ("setty", "human", "Setty HSPC — CD34+ bone marrow"),
    ("hESC_GSE144024", "human", "hESC embryonic stem cells"),
]


def lorentz_norm(latent):
    """Compute the Lorentz norm r = arccosh(sqrt(1 + |z|^2)).

    In the Lorentz hyperboloid parameterisation, the pre-image of a
    vector z in R^d under ExpMap_0 has Lorentz norm:
        r(z) = arccosh(sqrt(1 + ||z||^2))
    """
    z_norm_sq = np.sum(latent ** 2, axis=1)
    return np.arccosh(np.sqrt(1.0 + z_norm_sq))


def root_cell_from_latent(latent, n_components=2):
    """Pick the cell with the smallest Lorentz norm as root.

    In hyperbolic geometry, the origin corresponds to the most
    undifferentiated state (stem/progenitor).
    """
    return int(np.argmin(lorentz_norm(latent)))


def scanpy_dpt(adata, root_cell):
    """Compute diffusion pseudotime with scanpy."""
    # Copy to avoid contaminating the upstream adata
    ad = adata.copy()
    sc.pp.pca(ad, n_comps=30)
    sc.pp.neighbors(ad, n_neighbors=15, use_rep="X_pca")
    sc.tl.diffmap(ad, n_comps=15)
    ad.uns["iroot"] = root_cell
    sc.tl.dpt(ad)
    return ad.obs["dpt_pseudotime"].values


def pca_distance_pseudotime(adata, root_cell):
    """PCA baseline: Euclidean distance from root cell in 30D PCA space."""
    ad = adata.copy()
    sc.pp.pca(ad, n_comps=30)
    X = ad.obsm["X_pca"]
    return np.linalg.norm(X - X[root_cell], axis=1)


def umap_distance_pseudotime(adata, root_cell):
    """UMAP baseline: Euclidean distance from root cell in UMAP 2D space."""
    ad = adata.copy()
    sc.pp.pca(ad, n_comps=30)
    sc.pp.neighbors(ad, n_neighbors=15, use_rep="X_pca")
    sc.tl.umap(ad, min_dist=0.3, n_components=2)
    X = ad.obsm["X_umap"]
    return np.linalg.norm(X - X[root_cell], axis=1)


def evaluate_pseudotime(pt, reference):
    """Compare a candidate pseudotime against reference (e.g., DPT)."""
    from scipy.stats import spearmanr, kendalltau
    # Remove NaN/inf
    mask = np.isfinite(pt) & np.isfinite(reference)
    pt_clean = pt[mask]
    ref_clean = reference[mask]
    if len(pt_clean) < 10:
        return {"spearman": np.nan, "kendall": np.nan, "rank_dist": np.nan}

    sp, _ = spearmanr(pt_clean, ref_clean)
    kt, _ = kendalltau(pt_clean, ref_clean)

    # Normalised mean absolute rank distance (lower = better)
    pt_rank = np.argsort(np.argsort(pt_clean))
    ref_rank = np.argsort(np.argsort(ref_clean))
    n = len(pt_rank)
    rank_dist = np.mean(np.abs(pt_rank - ref_rank)) / (n - 1)

    return {"spearman": sp, "kendall": kt, "rank_dist": rank_dist}


def run_on_dataset(name, organism, description):
    # Find dataset
    ds_files = discover_datasets()
    match = [f for f in ds_files if name in os.path.basename(f)]
    if not match:
        logger.warning("Dataset %s not found", name)
        return None

    filepath = match[0]
    logger.info("=" * 60)
    logger.info("Dataset: %s (%s)", name, description)
    logger.info("=" * 60)

    adata1 = load_and_preprocess(filepath)
    labels, n_clusters = get_labels(adata1)
    logger.info("Cells: %d, Clusters: %d", adata1.n_obs, n_clusters)

    # 1. Train GAHIB
    logger.info("Training GAHIB...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAHIB(
        adata1, layer="counts",
        recon=1.0, irecon=0.5, lorentz=5.0, beta=0.1,
        encoder_type="graph", graph_type="GAT",
        hidden_dim=128, latent_dim=10, i_dim=2,
        lr=1e-4, loss_type="nb", device=device, random_seed=42,
    )
    model.fit(epochs=EPOCHS, patience=30, early_stop=True,
              compute_metrics=False)

    # 2. GAHIB Lorentz-norm pseudotime
    latent = model.get_latent()
    gahib_pt = lorentz_norm(latent)
    root = root_cell_from_latent(latent)
    logger.info("Root cell: %d (Lorentz norm=%.3f)", root, gahib_pt[root])

    # Pre-compute a 2D UMAP of the preprocessed data for visualisation
    ad_viz = adata1.copy()
    sc.pp.pca(ad_viz, n_comps=30)
    sc.pp.neighbors(ad_viz, n_neighbors=15, use_rep="X_pca")
    sc.tl.umap(ad_viz, min_dist=0.3, n_components=2)
    umap_2d = ad_viz.obsm["X_umap"].astype(np.float32)

    # 3. Reference: scanpy DPT
    logger.info("Computing scanpy DPT...")
    dpt = scanpy_dpt(adata1, root_cell=root)

    # 4. Baselines
    logger.info("Computing PCA baseline...")
    pca_pt = pca_distance_pseudotime(adata1, root_cell=root)

    logger.info("Computing UMAP baseline...")
    umap_pt = umap_distance_pseudotime(adata1, root_cell=root)

    # 5. Evaluate all vs DPT reference
    results = []
    for method_name, pt in [
        ("GAHIB (Lorentz)", gahib_pt),
        ("PCA distance", pca_pt),
        ("UMAP distance", umap_pt),
    ]:
        r = evaluate_pseudotime(pt, dpt)
        r["method"] = method_name
        r["dataset"] = name
        r["organism"] = organism
        r["n_cells"] = adata1.n_obs
        results.append(r)
        logger.info("  %s: Spearman=%.3f, Kendall=%.3f, RankDist=%.3f",
                    method_name, r["spearman"], r["kendall"],
                    r["rank_dist"])

    # Also check GAHIB vs PCA agreement (both pseudotimes, not DPT)
    r = evaluate_pseudotime(gahib_pt, pca_pt)
    r["method"] = "GAHIB vs PCA"
    r["dataset"] = name + "_pca_ref"
    r["organism"] = organism
    r["n_cells"] = adata1.n_obs
    results.append(r)

    # Save per-dataset details
    df = pd.DataFrame(results)
    csv = TABLES_DIR / f"traj_{name}_df.csv"
    df.to_csv(csv, index=False)
    logger.info("Saved: %s", csv)

    # 6. Gene-correlation biological validation of GAHIB pseudotime.
    #    For each HVG compute Pearson r(expression_i, gahib_pt).
    #    Genes with strongly negative r behave as early/progenitor-like
    #    (expression decreases along pseudotime); genes with strongly
    #    positive r behave as lineage-commitment genes.
    logger.info("Computing per-gene Pearson correlations with GAHIB pt ...")
    import scipy.sparse as sp
    X = adata1.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    var_names = np.asarray(adata1.var_names.astype(str))

    # Compute Pearson correlation per gene vs gahib_pt (vectorised)
    pt = np.asarray(gahib_pt, dtype=np.float32)
    pt_c = pt - pt.mean()
    pt_n = pt_c / (np.linalg.norm(pt_c) + 1e-12)
    Xc = X - X.mean(axis=0, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=0, keepdims=True) + 1e-12)
    r_vals = (Xn * pt_n[:, None]).sum(axis=0)
    n_cells = pt.shape[0]
    t_vals = r_vals * np.sqrt(n_cells - 2) / np.sqrt(np.maximum(
        1.0 - r_vals ** 2, 1e-12))
    from scipy.stats import t as _t
    p_vals = 2.0 * _t.sf(np.abs(t_vals), df=n_cells - 2)

    gc_df = pd.DataFrame({
        "gene": var_names,
        "pearson_r": r_vals,
        "pvalue": p_vals,
    })
    gc_df["abs_r"] = gc_df["pearson_r"].abs()
    gc_df = gc_df.sort_values("pearson_r", ascending=False).reset_index(drop=True)
    gc_csv = TABLES_DIR / f"traj_{name}_gene_corr.csv"
    gc_df.to_csv(gc_csv, index=False)
    logger.info("Saved gene correlations: %s", gc_csv)

    # Save expression of top-k positive and negative genes for scatter reuse
    top_k = 10
    pos_idx = np.argsort(r_vals)[::-1][:top_k]
    neg_idx = np.argsort(r_vals)[:top_k]
    keep_idx = np.unique(np.concatenate([pos_idx, neg_idx]))

    # Also save the raw pseudotime arrays for figure regeneration
    np.savez(
        TABLES_DIR / f"traj_{name}_pts.npz",
        gahib_pt=gahib_pt, dpt=dpt, pca_pt=pca_pt, umap_pt=umap_pt,
        labels=labels.astype(str), root=root,
        umap_2d=umap_2d,
        top_gene_expr=X[:, keep_idx].astype(np.float32),
        top_gene_names=var_names[keep_idx],
        top_gene_r=r_vals[keep_idx].astype(np.float32),
    )

    del model
    torch.cuda.empty_cache()
    return df


def main():
    logger.info("\n%s", "#" * 60)
    logger.info("  Pseudotime / Trajectory Validation")
    logger.info("%s\n", "#" * 60)

    all_results = []
    for name, organism, desc in TARGET_DATASETS:
        try:
            df = run_on_dataset(name, organism, desc)
            if df is not None:
                all_results.append(df)
        except Exception as e:
            logger.error("FAILED: %s: %s", name, e)
            traceback.print_exc()

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_csv = RESULTS_DIR / "trajectory_summary.csv"
        combined.to_csv(combined_csv, index=False)
        logger.info("\nSaved combined summary: %s", combined_csv)


if __name__ == "__main__":
    main()
