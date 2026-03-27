#!/usr/bin/env python3
"""
Experiment 10: Model Interpretation & Biological Validation
=============================================================
Post-hoc interpretation and downstream biovalidation of the trained GAHIB
model to validate that each architectural component (GAT, IB, Lorentz
geometry) learns biologically meaningful structure.

**Model Interpretation (4 axes):**
  1. GAT Attention — cell-type attention patterns & homophily
  2. Information Bottleneck — 2D bottleneck structure & retention
  3. Hyperbolic Geometry — Lorentz norms (hierarchy) & Poincaré projection
  4. Gene Attribution — Decoder Jacobian per latent dimension

**Biological Validation (5 axes):**
  5. Gene Program Discovery — enrichment of top-attributed genes
  6. Stemness–Hierarchy Correlation — Lorentz norms vs differentiation
  7. Marker Gene Recovery — overlap with DE markers per cluster
  8. Latent Traversal — in-silico perturbation gene response curves
  9. Reconstruction Quality — per-cell / per-gene / per-type fidelity

Runs on the same 12 datasets as all other experiments.
Produces per-dataset figures (11 types) + cross-dataset summary.
"""

import sys
import os
import gc
import logging
import json
import traceback

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.exp_utils import (
    discover_datasets, get_labels, load_and_preprocess, get_done_datasets,
)
from gahib import GAHIB
from gahib.interpretation import (
    run_interpretation,
    run_biovalidation,
    build_hyperbolic_hierarchy,
    InterpretationResult,
)
from gahib.viz.interpretation import (
    generate_interpretation_figures,
    fig_interpretation_summary,
    fig_biovalidation_summary,
)
from gahib.viz import style as S

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Configuration (matches all other experiments) ──
EPOCHS = 200
EXPERIMENT = "interpretation"
PREFIX = "interp"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "GAHIB_results", EXPERIMENT)
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
ARRAYS_DIR = os.path.join(RESULTS_DIR, "arrays")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(ARRAYS_DIR, exist_ok=True)

GAHIB_CONFIG = dict(
    recon=1.0, irecon=0.5, lorentz=5.0, beta=0.1,
    encoder_type="graph", graph_type="GAT",
)

# Completion marker: the biovalidation summary JSON
DONE_MARKER = "_bioval.json"


def run_single_dataset(filepath: str, dataset_name: str) -> dict:
    """Train GAHIB, run interpretation + biovalidation on one dataset.

    Returns a summary dict for the cross-dataset summary figures.
    """
    logger.info("=" * 60)
    logger.info("Dataset: %s", dataset_name)
    logger.info("=" * 60)

    # ── 1. Preprocess ──
    adata1 = load_and_preprocess(filepath)
    labels, n_clusters = get_labels(adata1)
    gene_names = np.array(adata1.var_names)

    # ── 2. Train GAHIB ──
    logger.info("Training GAHIB (epochs=%d)...", EPOCHS)
    m = GAHIB(
        adata1, layer="counts",
        hidden_dim=128, latent_dim=10, i_dim=2,
        batch_size=128, random_seed=42,
        **GAHIB_CONFIG,
    )
    m.fit(epochs=EPOCHS, patience=30, early_stop=True, compute_metrics=False)
    logger.info("  Training complete: %d epochs, %.1fs",
                m.actual_epochs, m.train_time)

    # ── 3. Model interpretation (4 axes) ──
    result = run_interpretation(
        m, labels, dataset_name,
        gene_names=gene_names,
        n_jacobian_samples=200,
    )

    # ── 4. Biological validation (5 axes) ──
    result = run_biovalidation(
        m, adata1, labels, result,
        run_enrichment=True,
        run_traversal=True,
    )

    # ── 5. Compute UMAP for overlay plots ──
    umap_coords = None
    try:
        import scanpy as sc
        import anndata as ad
        latent = m.get_latent()
        adata_latent = ad.AnnData(X=latent)
        sc.pp.neighbors(adata_latent, use_rep="X", n_neighbors=15)
        sc.tl.umap(adata_latent)
        umap_coords = adata_latent.obsm["X_umap"]
    except Exception as e:
        logger.warning("  UMAP computation failed: %s", e)

    # ── 6. Generate all per-dataset figures (11 types) ──
    ds_fig_dir = os.path.join(FIGURES_DIR, dataset_name)
    saved = generate_interpretation_figures(result, ds_fig_dir, umap_coords)
    logger.info("  Saved %d figure files to %s", len(saved), ds_fig_dir)

    # ── 6b. Save per-cell arrays for themed cross-dataset figures ──
    _save_percell_arrays(result, m, umap_coords, dataset_name)

    # ── 7. Save tables ──
    _save_all_tables(result, dataset_name)

    # ── 8. Collect summary stats ──
    n_active = int(np.sum(result.dim_variance > 0.01)) if result.dim_variance is not None else 0

    # Hierarchy score
    hierarchy = build_hyperbolic_hierarchy(
        result.z_manifold, labels, result.lorentz_norms,
    ) if result.z_manifold is not None else {"hierarchy_score": 0.0}

    # Mean marker overlap
    mean_marker = 0.0
    if result.marker_overlap:
        fracs = [v["overlap_fraction"] for v in result.marker_overlap.values()]
        mean_marker = float(np.mean(fracs)) if fracs else 0.0

    summary = {
        "dataset": dataset_name,
        "n_cells": adata1.n_obs,
        "n_clusters": n_clusters,
        "train_time": m.train_time,
        "actual_epochs": m.actual_epochs,
        # Interpretation
        "attn_homophily": result.attn_homophily or 0.0,
        "mean_lorentz_norm": float(result.lorentz_norms.mean()) if result.lorentz_norms is not None else 0.0,
        "mean_ib_retention": float(result.ib_retention.mean()) if result.ib_retention is not None else 0.0,
        "n_active_dims": n_active,
        # Biovalidation
        "stemness_corr": result.stemness_norm_corr if result.stemness_norm_corr is not None else 0.0,
        "stemness_pval": result.stemness_norm_pval if result.stemness_norm_pval is not None else 1.0,
        "mean_marker_overlap": mean_marker,
        "hierarchy_score": hierarchy["hierarchy_score"],
        "mean_recon_mse": float(result.recon_per_cell.mean()) if result.recon_per_cell is not None else 0.0,
        "n_enriched_terms": sum(len(v) for v in result.enrichment_results.values()) if result.enrichment_results else 0,
    }

    # Save completion marker
    with open(os.path.join(TABLES_DIR, f"{PREFIX}_{dataset_name}{DONE_MARKER}"), "w") as f:
        json.dump(summary, f, indent=2)

    # Cleanup
    del m, adata1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def _save_all_tables(result: InterpretationResult, dataset_name: str):
    """Save all interpretation + biovalidation tables to CSV/JSON."""

    # --- Interpretation tables ---

    # Lorentz norms per cell type
    if result.lorentz_norms is not None:
        rows = []
        for lbl in result.label_names:
            mask = result.labels == lbl
            norms = result.lorentz_norms[mask]
            rows.append({
                "cell_type": lbl,
                "mean_norm": norms.mean(),
                "std_norm": norms.std(),
                "min_norm": norms.min(),
                "max_norm": norms.max(),
                "n_cells": mask.sum(),
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(TABLES_DIR, f"{PREFIX}_{dataset_name}_lorentz.csv"),
            index=False)

    # IB retention per cell type
    if result.ib_retention is not None:
        rows = []
        for lbl in result.label_names:
            mask = result.labels == lbl
            ret = result.ib_retention[mask]
            rows.append({
                "cell_type": lbl,
                "mean_retention": ret.mean(),
                "std_retention": ret.std(),
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(TABLES_DIR, f"{PREFIX}_{dataset_name}_ib.csv"),
            index=False)

    # Top genes per dimension
    if result.top_genes_per_dim:
        rows = []
        for dim, genes in result.top_genes_per_dim.items():
            for rank, gene in enumerate(genes):
                score = 0.0
                if result.gene_names is not None:
                    matches = np.where(result.gene_names == gene)[0]
                    if len(matches) > 0:
                        score = float(result.gene_scores[matches[0], dim])
                rows.append({
                    "dimension": dim, "rank": rank + 1,
                    "gene": gene, "score": score,
                })
        pd.DataFrame(rows).to_csv(
            os.path.join(TABLES_DIR, f"{PREFIX}_{dataset_name}_genes.csv"),
            index=False)

    # Attention metadata
    if result.attn_homophily is not None:
        with open(os.path.join(TABLES_DIR, f"{PREFIX}_{dataset_name}_attention.json"), "w") as f:
            json.dump({
                "homophily_top10pct": result.attn_homophily,
                "n_edges": result.edge_index.shape[1] if result.edge_index is not None else 0,
                "n_gat_layers": len(result.attention_weights) if result.attention_weights else 0,
            }, f, indent=2)

    # Dimension variance
    if result.dim_variance is not None:
        pd.DataFrame({
            "dimension": range(len(result.dim_variance)),
            "variance": result.dim_variance,
        }).to_csv(
            os.path.join(TABLES_DIR, f"{PREFIX}_{dataset_name}_dims.csv"),
            index=False)

    # --- Biovalidation tables ---

    # Stemness correlation
    if result.stemness_norm_corr is not None:
        with open(os.path.join(TABLES_DIR, f"{PREFIX}_{dataset_name}_stemness.json"), "w") as f:
            json.dump({
                "spearman_corr": result.stemness_norm_corr,
                "pvalue": result.stemness_norm_pval,
            }, f, indent=2)

    # Marker gene overlap
    if result.marker_overlap:
        rows = []
        for cluster, data in result.marker_overlap.items():
            rows.append({
                "cluster": cluster,
                "n_de_markers": data["n_de_markers"],
                "overlap_count": data["overlap_count"],
                "overlap_fraction": data["overlap_fraction"],
                "best_dim": data["best_matching_dim"],
                "best_dim_overlap": data["best_dim_overlap"],
                "overlap_genes": "; ".join(data.get("overlap_genes", [])),
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(TABLES_DIR, f"{PREFIX}_{dataset_name}_markers.csv"),
            index=False)

    # Gene enrichment results
    if result.enrichment_results:
        rows = []
        for dim, hits in result.enrichment_results.items():
            for h in hits:
                rows.append({
                    "dimension": dim,
                    "term": h.get("term", ""),
                    "pvalue": h.get("pvalue", 1.0),
                    "source": h.get("source", ""),
                    "genes": h.get("genes", ""),
                })
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(TABLES_DIR, f"{PREFIX}_{dataset_name}_enrichment.csv"),
                index=False)

    # Reconstruction quality per type
    if result.recon_per_type:
        rows = [{"cell_type": k, "mean_mse": v} for k, v in result.recon_per_type.items()]
        pd.DataFrame(rows).to_csv(
            os.path.join(TABLES_DIR, f"{PREFIX}_{dataset_name}_recon.csv"),
            index=False)

    # Hyperbolic hierarchy
    if result.hyp_dist_matrix is not None:
        pd.DataFrame(
            result.hyp_dist_matrix,
            index=result.hyp_dist_labels,
            columns=result.hyp_dist_labels,
        ).to_csv(
            os.path.join(TABLES_DIR, f"{PREFIX}_{dataset_name}_hypdist.csv"))


def _load_existing_summary(dataset_name: str) -> dict:
    """Load summary from a previously completed dataset."""
    bioval_path = os.path.join(TABLES_DIR, f"{PREFIX}_{dataset_name}{DONE_MARKER}")
    if os.path.exists(bioval_path):
        with open(bioval_path) as f:
            return json.load(f)
    return None


def _save_percell_arrays(result, model, umap_coords, dataset_name):
    """Save per-cell arrays for cross-dataset themed figures."""
    save_dict = {}

    if umap_coords is not None:
        save_dict["umap_coords"] = umap_coords.astype(np.float32)
    if result.poincare_coords is not None:
        save_dict["poincare_coords"] = result.poincare_coords.astype(np.float32)
    if result.lorentz_norms is not None:
        save_dict["lorentz_norms"] = result.lorentz_norms.astype(np.float32)
    save_dict["labels"] = result.labels
    if result.label_names is not None:
        save_dict["label_names"] = np.array(result.label_names)
    if result.q_z is not None:
        save_dict["q_z"] = result.q_z.astype(np.float32)
    if result.dim_variance is not None:
        save_dict["dim_variance"] = result.dim_variance.astype(np.float32)
    if result.stemness_scores is not None:
        save_dict["stemness_scores"] = result.stemness_scores.astype(np.float32)
    if result.gene_scores is not None:
        save_dict["gene_scores"] = result.gene_scores
    if result.gene_names is not None:
        save_dict["gene_names"] = result.gene_names
    if result.le is not None:
        save_dict["le"] = result.le.astype(np.float32)

    # Top-K gene expression columns for themed gene-on-UMAP figures
    if result.gene_scores is not None and result.gene_names is not None:
        K = 20
        overall = result.gene_scores.mean(axis=1)
        top_idx = np.argsort(overall)[-K:][::-1]
        X_norm = model.X_norm
        if hasattr(X_norm, "toarray"):
            X_norm = X_norm.toarray()
        save_dict["top_gene_expr"] = np.asarray(
            X_norm[:, top_idx], dtype=np.float32)
        save_dict["top_gene_expr_names"] = result.gene_names[top_idx]

    path = os.path.join(ARRAYS_DIR, f"{dataset_name}.npz")
    np.savez_compressed(path, **save_dict)
    logger.info("  Saved per-cell arrays to %s", path)


def main():
    """Run interpretation + biovalidation on all 12 datasets."""

    logger.info("GAHIB Interpretation & Biovalidation Experiment")
    logger.info("=" * 60)

    datasets = discover_datasets()
    if not datasets:
        logger.error("No datasets found. Set GAHIB_DATASET_DIRS env var.")
        return

    logger.info("Found %d datasets", len(datasets))

    # Check completion markers
    done_datasets = set()
    for f in os.listdir(TABLES_DIR):
        if f.startswith(PREFIX) and f.endswith(DONE_MARKER):
            name = f.replace(f"{PREFIX}_", "").replace(DONE_MARKER, "")
            done_datasets.add(name)

    summaries = []

    for filepath in datasets:
        dataset_name = os.path.basename(filepath).replace(".h5ad", "")
        for suffix in ["_processed", "_filtered"]:
            dataset_name = dataset_name.replace(suffix, "")

        if dataset_name in done_datasets:
            logger.info("Skipping %s (already done)", dataset_name)
            existing = _load_existing_summary(dataset_name)
            if existing:
                summaries.append(existing)
            continue

        try:
            summary = run_single_dataset(filepath, dataset_name)
            summaries.append(summary)
        except Exception as e:
            logger.error("FAILED on %s: %s", dataset_name, e)
            traceback.print_exc()
            continue

    # ── Cross-dataset summary figures ──
    if len(summaries) >= 2:
        logger.info("\nGenerating cross-dataset summaries...")

        # Interpretation summary (4-panel)
        fig = fig_interpretation_summary(
            dataset_names=[s["dataset"] for s in summaries],
            homophily_scores=[s.get("attn_homophily", 0) for s in summaries],
            mean_lorentz_norms=[s.get("mean_lorentz_norm", 0) for s in summaries],
            mean_ib_retention=[s.get("mean_ib_retention", 0) for s in summaries],
            n_active_dims=[s.get("n_active_dims", 0) for s in summaries],
        )
        S.save_figure(fig, os.path.join(FIGURES_DIR, "fig_interpretation_summary.png"))
        plt.close(fig)

        # Biovalidation summary (6-panel)
        fig = fig_biovalidation_summary(
            dataset_names=[s["dataset"] for s in summaries],
            stemness_corrs=[s.get("stemness_corr", 0) for s in summaries],
            mean_marker_overlap=[s.get("mean_marker_overlap", 0) for s in summaries],
            hierarchy_scores=[s.get("hierarchy_score", 0) for s in summaries],
            mean_recon_mse=[s.get("mean_recon_mse", 0) for s in summaries],
            homophily_scores=[s.get("attn_homophily", 0) for s in summaries],
            mean_lorentz_norms=[s.get("mean_lorentz_norm", 0) for s in summaries],
        )
        S.save_figure(fig, os.path.join(FIGURES_DIR, "fig_biovalidation_summary.png"))
        plt.close(fig)

        # Save combined summary table
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(os.path.join(TABLES_DIR, f"{PREFIX}_summary.csv"), index=False)
        logger.info("Summary saved to %s", TABLES_DIR)

    # ── Themed cross-dataset figures (grouped by interpretation theme) ──
    from gahib.viz.interpretation import generate_themed_figures
    from experiments.exp_utils import SELECTED_DATASETS
    themed_saved = generate_themed_figures(
        ARRAYS_DIR, FIGURES_DIR, dataset_order=SELECTED_DATASETS,
    )
    if themed_saved:
        logger.info("Themed figures: %d files saved", len(themed_saved))

    # ── Print final report ──
    logger.info("\n" + "=" * 60)
    logger.info("INTERPRETATION & BIOVALIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Datasets processed: %d", len(summaries))
    logger.info("Figures: %s", FIGURES_DIR)
    logger.info("Tables: %s", TABLES_DIR)

    if summaries:
        logger.info("\nCross-dataset highlights:")
        corrs = [s.get("stemness_corr", 0) for s in summaries]
        homos = [s.get("attn_homophily", 0) for s in summaries]
        markers = [s.get("mean_marker_overlap", 0) for s in summaries]
        logger.info("  Mean stemness-norm correlation: %.3f", np.mean(corrs))
        logger.info("  Mean attention homophily:       %.3f", np.mean(homos))
        logger.info("  Mean marker gene recovery:      %.3f", np.mean(markers))

    logger.info("\nPer-dataset figure types (10 per dataset):")
    logger.info("  1. fig_bottleneck   — IB 2D scatter + dimension analysis")
    logger.info("  2. fig_hyperbolic   — Poincaré disk + Lorentz norms")
    logger.info("  3. fig_attention    — Cell-type attention heatmap")
    logger.info("  4. fig_genes        — Decoder Jacobian gene attribution")
    logger.info("  5. fig_stemness     — Stemness–hierarchy correlation")
    logger.info("  6. fig_enrichment   — Gene program enrichment dot plot")
    logger.info("  7. fig_traversal    — Latent dimension gene response")
    logger.info("  8. fig_recon        — Reconstruction quality analysis")
    logger.info("  9. fig_hierarchy    — Hyperbolic distance matrix")
    logger.info("  10. fig_markers     — DE marker gene recovery")
    logger.info("  + 2 cross-dataset summary figures")
    logger.info("  + 8 themed cross-dataset figures (embedding, Poincaré,")
    logger.info("    latent dims × 3, gene expression × 3)")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.apply_style()
    main()
