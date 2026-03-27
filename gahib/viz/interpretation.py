# ============================================================================
# interpretation.py - Visualization functions for GAHIB model interpretation
# ============================================================================
"""
Publication-quality figures for interpreting GAHIB model components.

Produces 4 main figure types:
  Fig 1: Bottleneck & Latent Structure (IB 2D scatter + dimension utilization)
  Fig 2: Hyperbolic Geometry (Poincaré disk + Lorentz norm distributions)
  Fig 3: GAT Attention (cell-type attention heatmap + homophily)
  Fig 4: Gene Attribution (decoder Jacobian heatmap)

All figures use the centralized style system (gahib.viz.style).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from . import style as S

# Interpretation-specific colour ramp for continuous values
_NORM_CMAP = "viridis"
_CLUSTER_CMAP = "tab20"
_DIVERGING_CMAP = "RdBu_r"
_GENE_CMAP = "YlOrRd"

# Layout rects for interpretation figures
RECT_SCATTER_2x2 = [0.08, 0.08, 0.88, 0.84]
RECT_HEATMAP_WIDE = [0.14, 0.25, 0.80, 0.65]


# ============================================================================
# Colour helpers
# ============================================================================

def _cluster_palette(labels: np.ndarray) -> Tuple[Dict[str, str], np.ndarray]:
    """Generate a colour palette for cluster labels and return RGBA array."""
    unique = sorted(np.unique(labels).tolist())
    n = len(unique)
    cmap = plt.get_cmap(_CLUSTER_CMAP, max(n, 20))
    palette = {lbl: cmap(i / max(n - 1, 1)) for i, lbl in enumerate(unique)}
    colors = np.array([palette[l] for l in labels])
    return palette, colors


# ============================================================================
# Figure 1: Information Bottleneck & Latent Structure
# ============================================================================

def fig_bottleneck_analysis(
    le: np.ndarray,
    q_m: np.ndarray,
    q_s: np.ndarray,
    ib_retention: np.ndarray,
    labels: np.ndarray,
    dataset_name: str = "",
) -> plt.Figure:
    """4-panel figure: (a) IB 2D scatter, (b) IB retention per type,
    (c) dimension variance, (d) dimension utilization."""

    S.apply_style()
    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, S.FIG_WIDTH_IN * 0.7))
    axes = S.grid_of_axes(fig, 2, 2, RECT_SCATTER_2x2, hgap=0.10, wgap=0.10)
    palette, colors = _cluster_palette(labels)
    unique_labels = sorted(np.unique(labels).tolist())

    # (a) Bottleneck 2D scatter (i_dim=2)
    ax = axes[0][0]
    if le.shape[1] >= 2:
        ax.scatter(le[:, 0], le[:, 1], c=colors, s=4, alpha=0.6, rasterized=True)
        ax.set_xlabel("IB dim 1", fontsize=S.FS_AXIS)
        ax.set_ylabel("IB dim 2", fontsize=S.FS_AXIS)
    else:
        ax.hist(le[:, 0], bins=50, color="#0072B2", alpha=0.7)
        ax.set_xlabel("IB dim 1", fontsize=S.FS_AXIS)
    ax.set_title("Information Bottleneck", fontsize=S.FS_TITLE)
    S.add_panel_label(ax, "a")

    # (b) IB retention per cell type
    ax = axes[0][1]
    type_retention = []
    for lbl in unique_labels:
        mask = labels == lbl
        type_retention.append(ib_retention[mask].mean())
    y_pos = np.arange(len(unique_labels))
    bars = ax.barh(y_pos, type_retention, color="#D55E00", alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(unique_labels, fontsize=S.FS_SMALL)
    ax.set_xlabel("MSE(z, z')", fontsize=S.FS_AXIS)
    ax.set_title("IB Retention per Type", fontsize=S.FS_TITLE)
    ax.invert_yaxis()
    S.add_panel_label(ax, "b")

    # (c) Dimension variance (posterior means)
    ax = axes[1][0]
    dim_var = np.var(q_m, axis=0)
    dim_idx = np.arange(len(dim_var))
    sorted_order = np.argsort(dim_var)[::-1]
    ax.bar(dim_idx, dim_var[sorted_order], color="#0072B2", alpha=0.8)
    ax.set_xlabel("Latent dimension (sorted)", fontsize=S.FS_AXIS)
    ax.set_ylabel("Variance", fontsize=S.FS_AXIS)
    ax.set_title("Dimension Variance", fontsize=S.FS_TITLE)
    ax.set_xticks(dim_idx)
    ax.set_xticklabels(sorted_order, fontsize=S.FS_SMALL)
    S.add_panel_label(ax, "c")

    # (d) Dimension utilization
    ax = axes[1][1]
    import torch
    mean_post_var = np.mean(
        torch.nn.functional.softplus(torch.tensor(q_s)).numpy() ** 2, axis=0
    )
    utilization = dim_var / (mean_post_var + 1e-8)
    ax.bar(dim_idx, utilization[sorted_order], color="#009E73", alpha=0.8)
    ax.set_xlabel("Latent dimension (sorted)", fontsize=S.FS_AXIS)
    ax.set_ylabel("Utilization ratio", fontsize=S.FS_AXIS)
    ax.set_title("Dimension Utilization", fontsize=S.FS_TITLE)
    ax.set_xticks(dim_idx)
    ax.set_xticklabels(sorted_order, fontsize=S.FS_SMALL)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    S.add_panel_label(ax, "d")

    if dataset_name:
        fig.suptitle(dataset_name, fontsize=S.FS_TITLE + 1, y=1.02)

    return fig


# ============================================================================
# Figure 2: Hyperbolic Geometry
# ============================================================================

def fig_hyperbolic_geometry(
    poincare_coords: np.ndarray,
    lorentz_norms: np.ndarray,
    labels: np.ndarray,
    dataset_name: str = "",
    umap_coords: Optional[np.ndarray] = None,
) -> plt.Figure:
    """4-panel figure: (a) Poincaré disk (2D PCA), (b) Lorentz norms per type,
    (c) Lorentz norm vs UMAP, (d) norm distribution."""

    S.apply_style()
    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, S.FIG_WIDTH_IN * 0.7))
    axes = S.grid_of_axes(fig, 2, 2, RECT_SCATTER_2x2, hgap=0.10, wgap=0.10)
    palette, colors = _cluster_palette(labels)
    unique_labels = sorted(np.unique(labels).tolist())

    # (a) Poincaré disk projection (PCA to 2D if needed)
    ax = axes[0][0]
    if poincare_coords.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(poincare_coords)
    else:
        coords_2d = poincare_coords

    # Draw unit disk boundary
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5, alpha=0.3)
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=colors, s=4,
               alpha=0.6, rasterized=True)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel("Poincaré dim 1", fontsize=S.FS_AXIS)
    ax.set_ylabel("Poincaré dim 2", fontsize=S.FS_AXIS)
    ax.set_title("Poincaré Disk Projection", fontsize=S.FS_TITLE)
    S.add_panel_label(ax, "a")

    # (b) Lorentz norm per cell type (box plot)
    ax = axes[0][1]
    type_norms = [lorentz_norms[labels == lbl] for lbl in unique_labels]
    bp = ax.boxplot(type_norms, labels=unique_labels, vert=True, patch_artist=True,
                    widths=0.6, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.0))
    for i, patch in enumerate(bp['boxes']):
        c = palette[unique_labels[i]]
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Lorentz norm", fontsize=S.FS_AXIS)
    ax.set_title("Hyperbolic Radius per Type", fontsize=S.FS_TITLE)
    ax.tick_params(axis='x', rotation=45, labelsize=S.FS_SMALL)
    S.add_panel_label(ax, "b")

    # (c) Scatter: Lorentz norm coloured on UMAP (or PCA of latent)
    ax = axes[1][0]
    if umap_coords is not None and umap_coords.shape[1] >= 2:
        sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                        c=lorentz_norms, cmap=_NORM_CMAP, s=4,
                        alpha=0.6, rasterized=True)
        ax.set_xlabel("UMAP 1", fontsize=S.FS_AXIS)
        ax.set_ylabel("UMAP 2", fontsize=S.FS_AXIS)
    else:
        # Fall back to PCA of poincaré coords
        sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                        c=lorentz_norms, cmap=_NORM_CMAP, s=4,
                        alpha=0.6, rasterized=True)
        ax.set_xlabel("Poincaré dim 1", fontsize=S.FS_AXIS)
        ax.set_ylabel("Poincaré dim 2", fontsize=S.FS_AXIS)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Lorentz norm")
    ax.set_title("Hyperbolic Depth Map", fontsize=S.FS_TITLE)
    S.add_panel_label(ax, "c")

    # (d) Lorentz norm histogram by type (stacked)
    ax = axes[1][1]
    for lbl in unique_labels:
        mask = labels == lbl
        ax.hist(lorentz_norms[mask], bins=30, alpha=0.5, label=lbl,
                color=palette[lbl], density=True)
    ax.set_xlabel("Lorentz norm", fontsize=S.FS_AXIS)
    ax.set_ylabel("Density", fontsize=S.FS_AXIS)
    ax.set_title("Norm Distributions", fontsize=S.FS_TITLE)
    if len(unique_labels) <= 8:
        ax.legend(fontsize=S.FS_SMALL - 1, loc="upper right", ncol=1)
    S.add_panel_label(ax, "d")

    if dataset_name:
        fig.suptitle(dataset_name, fontsize=S.FS_TITLE + 1, y=1.02)

    return fig


# ============================================================================
# Figure 3: GAT Attention Analysis
# ============================================================================

def fig_attention_analysis(
    attn_type_matrix: np.ndarray,
    label_names: List[str],
    attn_homophily: float,
    edge_index: np.ndarray,
    attention_weights: np.ndarray,
    labels: np.ndarray,
    dataset_name: str = "",
) -> plt.Figure:
    """3-panel figure: (a) cell-type attention heatmap, (b) attention
    weight distribution, (c) homophily comparison (same vs cross-type)."""

    S.apply_style()
    ncols = 3
    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, S.FIG_WIDTH_IN * 0.38))
    axes = S.row_of_axes(fig, ncols, [0.06, 0.18, 0.90, 0.70], gap=0.08)

    # (a) Cell-type attention heatmap
    ax = axes[0]
    n_types = len(label_names)
    mat_norm = attn_type_matrix.copy()
    row_sums = mat_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat_norm /= row_sums  # row-normalize

    im = ax.imshow(mat_norm, cmap=_GENE_CMAP, aspect="auto", vmin=0)
    ax.set_xticks(range(n_types))
    ax.set_yticks(range(n_types))
    ax.set_xticklabels(label_names, fontsize=S.FS_SMALL, rotation=45, ha="right")
    ax.set_yticklabels(label_names, fontsize=S.FS_SMALL)
    ax.set_xlabel("Target type", fontsize=S.FS_AXIS)
    ax.set_ylabel("Source type", fontsize=S.FS_AXIS)
    ax.set_title("Attention Flow", fontsize=S.FS_TITLE)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    S.add_panel_label(ax, "a")

    # (b) Attention weight distribution (same-type vs cross-type)
    ax = axes[1]
    if attention_weights.ndim == 2:
        attn_flat = attention_weights.mean(axis=1)
    else:
        attn_flat = attention_weights
    src_labels = labels[edge_index[0]]
    dst_labels = labels[edge_index[1]]
    same_mask = src_labels == dst_labels
    ax.hist(attn_flat[same_mask], bins=50, alpha=0.6, label="Same type",
            color="#0072B2", density=True)
    ax.hist(attn_flat[~same_mask], bins=50, alpha=0.6, label="Cross type",
            color="#D55E00", density=True)
    ax.set_xlabel("Attention weight", fontsize=S.FS_AXIS)
    ax.set_ylabel("Density", fontsize=S.FS_AXIS)
    ax.set_title("Same vs Cross-Type", fontsize=S.FS_TITLE)
    ax.legend(fontsize=S.FS_SMALL)
    S.add_panel_label(ax, "b")

    # (c) Homophily bar
    ax = axes[2]
    categories = ["Same-type\n(top 10%)", "Random\nbaseline"]
    values = [attn_homophily, _random_homophily_baseline(labels)]
    bar_colors = ["#009E73", "#999999"]
    ax.bar(categories, values, color=bar_colors, alpha=0.8, width=0.5)
    ax.set_ylabel("Fraction", fontsize=S.FS_AXIS)
    ax.set_title("Attention Homophily", fontsize=S.FS_TITLE)
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=S.FS_SMALL)
    S.add_panel_label(ax, "c")

    if dataset_name:
        fig.suptitle(dataset_name, fontsize=S.FS_TITLE + 1, y=1.02)

    return fig


def _random_homophily_baseline(labels: np.ndarray) -> float:
    """Expected homophily under random pairing = sum(p_i^2)."""
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float((probs ** 2).sum())


# ============================================================================
# Figure 4: Gene Attribution (Decoder Jacobian)
# ============================================================================

def fig_gene_attribution(
    gene_scores: np.ndarray,
    gene_names: np.ndarray,
    top_k: int = 15,
    dataset_name: str = "",
) -> plt.Figure:
    """2-panel figure: (a) top genes per dimension heatmap,
    (b) overall gene importance bar chart."""

    S.apply_style()
    latent_dim = gene_scores.shape[1]

    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, S.FIG_WIDTH_IN * 0.55))
    axes = S.row_of_axes(fig, 2, [0.10, 0.12, 0.85, 0.78], gap=0.12,
                          widths=[2.0, 1.0])

    # (a) Top genes per dimension heatmap
    ax = axes[0]

    # For each dimension, find union of top-K genes
    all_top_idx = set()
    for d in range(latent_dim):
        top_idx = np.argsort(gene_scores[:, d])[-top_k:]
        all_top_idx.update(top_idx)
    all_top_idx = sorted(all_top_idx)

    sub_scores = gene_scores[all_top_idx]
    sub_names = gene_names[all_top_idx]

    # Normalize per-dimension for visibility
    col_max = sub_scores.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1
    sub_norm = sub_scores / col_max

    im = ax.imshow(sub_norm, cmap=_GENE_CMAP, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(latent_dim))
    ax.set_xticklabels([f"z{d}" for d in range(latent_dim)],
                       fontsize=S.FS_SMALL)
    ax.set_yticks(range(len(sub_names)))
    ax.set_yticklabels(sub_names, fontsize=max(5, S.FS_SMALL - 1))
    ax.set_xlabel("Latent dimension", fontsize=S.FS_AXIS)
    ax.set_ylabel("Gene", fontsize=S.FS_AXIS)
    ax.set_title("Top Gene Attribution per Dimension", fontsize=S.FS_TITLE)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Norm. score")
    S.add_panel_label(ax, "a")

    # (b) Overall gene importance (mean across dims)
    ax = axes[1]
    overall = gene_scores.mean(axis=1)
    top_overall_idx = np.argsort(overall)[-top_k:][::-1]
    top_overall_names = gene_names[top_overall_idx]
    top_overall_scores = overall[top_overall_idx]

    y_pos = np.arange(top_k)
    ax.barh(y_pos, top_overall_scores, color="#D55E00", alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_overall_names, fontsize=S.FS_SMALL)
    ax.set_xlabel("Mean |dg/dz|", fontsize=S.FS_AXIS)
    ax.set_title(f"Top-{top_k} Genes Overall", fontsize=S.FS_TITLE)
    ax.invert_yaxis()
    S.add_panel_label(ax, "b")

    if dataset_name:
        fig.suptitle(dataset_name, fontsize=S.FS_TITLE + 1, y=1.02)

    return fig


# ============================================================================
# Figure 5: Cross-Dataset Summary (aggregated interpretation)
# ============================================================================

def fig_interpretation_summary(
    dataset_names: List[str],
    homophily_scores: List[float],
    mean_lorentz_norms: List[float],
    mean_ib_retention: List[float],
    n_active_dims: List[int],
) -> plt.Figure:
    """4-panel summary across datasets: (a) attention homophily,
    (b) mean Lorentz norms, (c) IB retention, (d) active dimensions."""

    S.apply_style()
    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, S.FIG_WIDTH_IN * 0.7))
    axes = S.grid_of_axes(fig, 2, 2, RECT_SCATTER_2x2, hgap=0.12, wgap=0.10)
    n = len(dataset_names)
    x = np.arange(n)

    short_names = [d[:12] for d in dataset_names]

    # (a) Attention homophily
    ax = axes[0][0]
    ax.bar(x, homophily_scores, color="#009E73", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=S.FS_SMALL)
    ax.set_ylabel("Homophily", fontsize=S.FS_AXIS)
    ax.set_title("GAT Attention Homophily", fontsize=S.FS_TITLE)
    ax.set_ylim(0, 1)
    S.add_panel_label(ax, "a")

    # (b) Mean Lorentz norms
    ax = axes[0][1]
    ax.bar(x, mean_lorentz_norms, color="#0072B2", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=S.FS_SMALL)
    ax.set_ylabel("Mean norm", fontsize=S.FS_AXIS)
    ax.set_title("Hyperbolic Radius", fontsize=S.FS_TITLE)
    S.add_panel_label(ax, "b")

    # (c) IB retention
    ax = axes[1][0]
    ax.bar(x, mean_ib_retention, color="#D55E00", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=S.FS_SMALL)
    ax.set_ylabel("Mean MSE", fontsize=S.FS_AXIS)
    ax.set_title("Bottleneck Retention", fontsize=S.FS_TITLE)
    S.add_panel_label(ax, "c")

    # (d) Active dimensions
    ax = axes[1][1]
    ax.bar(x, n_active_dims, color="#CC79A7", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=S.FS_SMALL)
    ax.set_ylabel("Active dims", fontsize=S.FS_AXIS)
    ax.set_title("Dimension Utilization", fontsize=S.FS_TITLE)
    ax.set_ylim(0, max(n_active_dims) + 2)
    S.add_panel_label(ax, "d")

    fig.suptitle("Interpretation Summary", fontsize=S.FS_TITLE + 2, y=1.02)
    return fig


# ============================================================================
# Figure 6: Stemness–Hierarchy Correlation
# ============================================================================

def fig_stemness_hierarchy(
    stemness_scores: np.ndarray,
    lorentz_norms: np.ndarray,
    labels: np.ndarray,
    corr: float,
    pval: float,
    dataset_name: str = "",
    umap_coords: Optional[np.ndarray] = None,
) -> plt.Figure:
    """3-panel figure: (a) stemness vs Lorentz norm scatter,
    (b) stemness on UMAP, (c) per-type stemness vs norm box."""

    S.apply_style()
    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, S.FIG_WIDTH_IN * 0.38))
    axes = S.row_of_axes(fig, 3, [0.06, 0.18, 0.90, 0.70], gap=0.08)
    palette, colors = _cluster_palette(labels)
    unique_labels = sorted(np.unique(labels).tolist())

    # (a) Stemness vs Lorentz norm
    ax = axes[0]
    ax.scatter(lorentz_norms, stemness_scores, c=colors, s=3, alpha=0.4,
               rasterized=True)
    # Regression line
    z = np.polyfit(lorentz_norms, stemness_scores, 1)
    x_fit = np.linspace(lorentz_norms.min(), lorentz_norms.max(), 100)
    ax.plot(x_fit, np.polyval(z, x_fit), 'k--', linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Lorentz norm", fontsize=S.FS_AXIS)
    ax.set_ylabel("Stemness score", fontsize=S.FS_AXIS)
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
    ax.set_title(f"r={corr:.3f} ({sig})", fontsize=S.FS_TITLE)
    S.add_panel_label(ax, "a")

    # (b) Stemness on UMAP or Poincaré
    ax = axes[1]
    if umap_coords is not None and umap_coords.shape[1] >= 2:
        sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                        c=stemness_scores, cmap="coolwarm", s=3, alpha=0.5,
                        rasterized=True)
        ax.set_xlabel("UMAP 1", fontsize=S.FS_AXIS)
        ax.set_ylabel("UMAP 2", fontsize=S.FS_AXIS)
    else:
        sc = ax.scatter(lorentz_norms, stemness_scores,
                        c=stemness_scores, cmap="coolwarm", s=3, alpha=0.5,
                        rasterized=True)
        ax.set_xlabel("Lorentz norm", fontsize=S.FS_AXIS)
        ax.set_ylabel("Stemness", fontsize=S.FS_AXIS)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Stemness")
    ax.set_title("Stemness Map", fontsize=S.FS_TITLE)
    S.add_panel_label(ax, "b")

    # (c) Per-type: mean stemness vs mean norm
    ax = axes[2]
    type_stem = [stemness_scores[labels == l].mean() for l in unique_labels]
    type_norm = [lorentz_norms[labels == l].mean() for l in unique_labels]
    for i, lbl in enumerate(unique_labels):
        ax.scatter(type_norm[i], type_stem[i], color=palette[lbl], s=60,
                   zorder=3, edgecolors="black", linewidths=0.5)
        ax.annotate(lbl, (type_norm[i], type_stem[i]),
                    fontsize=S.FS_SMALL - 1, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Mean Lorentz norm", fontsize=S.FS_AXIS)
    ax.set_ylabel("Mean stemness", fontsize=S.FS_AXIS)
    ax.set_title("Type-Level Hierarchy", fontsize=S.FS_TITLE)
    S.add_panel_label(ax, "c")

    if dataset_name:
        fig.suptitle(dataset_name, fontsize=S.FS_TITLE + 1, y=1.02)

    return fig


# ============================================================================
# Figure 7: Gene Enrichment Summary
# ============================================================================

def fig_enrichment_summary(
    enrichment_results: Dict[int, List[Dict]],
    dataset_name: str = "",
    max_terms: int = 5,
) -> plt.Figure:
    """Dot plot of top enriched terms per latent dimension."""

    S.apply_style()

    # Collect all terms
    all_terms = []
    for d, hits in enrichment_results.items():
        for h in hits[:max_terms]:
            if h.get("pvalue", 1.0) < 0.05:
                all_terms.append({
                    "dim": d,
                    "term": h["term"][:40],
                    "pvalue": h["pvalue"],
                    "source": h.get("source", ""),
                })

    if not all_terms:
        # Empty figure with message
        fig = plt.figure(figsize=(S.FIG_WIDTH_IN, 2.0))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.text(0.5, 0.5, "No significant enrichment (p < 0.05)",
                ha="center", va="center", fontsize=S.FS_AXIS, transform=ax.transAxes)
        ax.axis("off")
        return fig

    import pandas as pd
    df = pd.DataFrame(all_terms)
    df["-log10(p)"] = -np.log10(df["pvalue"].clip(lower=1e-50))

    unique_terms = df["term"].unique()
    unique_dims = sorted(df["dim"].unique())

    fig_h = max(3.0, len(unique_terms) * 0.25 + 1.5)
    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, min(fig_h, S.FIG_HEIGHT_IN)))
    ax = fig.add_axes([0.40, 0.10, 0.55, 0.82])

    # Dot plot
    term_idx = {t: i for i, t in enumerate(unique_terms)}
    for _, row in df.iterrows():
        y = term_idx[row["term"]]
        x = row["dim"]
        size = min(row["-log10(p)"] * 15, 200)
        ax.scatter(x, y, s=size, c=[row["-log10(p)"]], cmap="Reds",
                   vmin=1.3, vmax=df["-log10(p)"].max(), edgecolors="black",
                   linewidths=0.3, zorder=3)

    ax.set_xticks(unique_dims)
    ax.set_xticklabels([f"z{d}" for d in unique_dims], fontsize=S.FS_SMALL)
    ax.set_yticks(range(len(unique_terms)))
    ax.set_yticklabels(unique_terms, fontsize=S.FS_SMALL - 1)
    ax.set_xlabel("Latent dimension", fontsize=S.FS_AXIS)
    ax.set_title("Gene Enrichment", fontsize=S.FS_TITLE)
    ax.invert_yaxis()

    if dataset_name:
        fig.suptitle(dataset_name, fontsize=S.FS_TITLE + 1, y=1.02)

    return fig


# ============================================================================
# Figure 8: Latent Traversal (Gene Response)
# ============================================================================

def fig_latent_traversal(
    traversal_responses: Dict[int, np.ndarray],
    gene_names: np.ndarray,
    top_k: int = 8,
    dims_to_show: Optional[List[int]] = None,
    dataset_name: str = "",
) -> plt.Figure:
    """Line plots showing gene expression changes along latent traversals.

    Shows top-K most responsive genes per selected dimension.
    """
    from ..interpretation import identify_responsive_genes

    S.apply_style()

    if dims_to_show is None:
        # Pick top-4 most responsive dimensions
        dim_ranges = {}
        for d, resp in traversal_responses.items():
            dim_ranges[d] = resp.max() - resp.min()
        dims_to_show = sorted(dim_ranges, key=dim_ranges.get, reverse=True)[:4]

    n_dims = len(dims_to_show)
    if n_dims == 0:
        fig = plt.figure(figsize=(S.FIG_WIDTH_IN, 2.0))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.text(0.5, 0.5, "No active dimensions for traversal",
                ha="center", va="center", fontsize=S.FS_AXIS, transform=ax.transAxes)
        ax.axis("off")
        return fig

    responsive = identify_responsive_genes(traversal_responses, gene_names, top_k=top_k)

    ncols = min(n_dims, 4)
    nrows = (n_dims + ncols - 1) // ncols
    fig_h = S.FIG_WIDTH_IN * 0.35 * nrows
    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, fig_h))
    axes = S.grid_of_axes(fig, nrows, ncols,
                          [0.08, 0.10, 0.88, 0.82],
                          hgap=0.12, wgap=0.08)

    panel_labels = "abcdefghijklmnop"

    for idx, d in enumerate(dims_to_show):
        r, c = idx // ncols, idx % ncols
        ax = axes[r][c] if nrows > 1 else axes[0][idx]
        resp = traversal_responses[d]
        n_steps = resp.shape[0]
        x_axis = np.linspace(-2, 2, n_steps)

        gene_list = responsive.get(d, [])
        cmap = plt.get_cmap("tab10", max(len(gene_list), 10))

        for gi, (gname, grange) in enumerate(gene_list[:top_k]):
            gene_idx = np.where(gene_names == gname)[0]
            if len(gene_idx) == 0:
                continue
            gene_idx = gene_idx[0]
            y = resp[:, gene_idx]
            ax.plot(x_axis, y, color=cmap(gi), linewidth=1.0, label=gname)

        ax.set_xlabel(f"z{d} (std)", fontsize=S.FS_SMALL)
        ax.set_ylabel("Predicted expr.", fontsize=S.FS_SMALL)
        ax.set_title(f"Dim z{d}", fontsize=S.FS_TITLE - 1)
        ax.legend(fontsize=S.FS_SMALL - 2, loc="upper left", ncol=2,
                  framealpha=0.5)
        S.add_panel_label(ax, panel_labels[idx])

    if dataset_name:
        fig.suptitle(f"Latent Traversal — {dataset_name}",
                     fontsize=S.FS_TITLE + 1, y=1.02)

    return fig


# ============================================================================
# Figure 9: Reconstruction Quality
# ============================================================================

def fig_reconstruction_quality(
    recon_per_cell: np.ndarray,
    recon_per_gene: np.ndarray,
    recon_per_type: Dict[str, float],
    labels: np.ndarray,
    gene_names: Optional[np.ndarray] = None,
    dataset_name: str = "",
) -> plt.Figure:
    """3-panel figure: (a) per-cell MSE by type, (b) per-type bar,
    (c) worst/best reconstructed genes."""

    S.apply_style()
    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, S.FIG_WIDTH_IN * 0.38))
    axes = S.row_of_axes(fig, 3, [0.06, 0.18, 0.90, 0.70], gap=0.08)
    palette, _ = _cluster_palette(labels)
    unique_labels = sorted(np.unique(labels).tolist())

    # (a) Per-cell MSE distribution by type
    ax = axes[0]
    type_mse = [recon_per_cell[labels == lbl] for lbl in unique_labels]
    bp = ax.boxplot(type_mse, labels=unique_labels, vert=True, patch_artist=True,
                    widths=0.6, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.0))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(palette[unique_labels[i]])
        patch.set_alpha(0.7)
    ax.set_ylabel("Recon. MSE", fontsize=S.FS_AXIS)
    ax.set_title("Per-Cell Quality", fontsize=S.FS_TITLE)
    ax.tick_params(axis='x', rotation=45, labelsize=S.FS_SMALL)
    S.add_panel_label(ax, "a")

    # (b) Per-type mean bar
    ax = axes[1]
    types_sorted = sorted(recon_per_type.keys(),
                          key=lambda k: recon_per_type[k])
    vals = [recon_per_type[t] for t in types_sorted]
    y_pos = np.arange(len(types_sorted))
    bar_colors = [palette.get(t, "#999999") for t in types_sorted]
    ax.barh(y_pos, vals, color=bar_colors, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(types_sorted, fontsize=S.FS_SMALL)
    ax.set_xlabel("Mean MSE", fontsize=S.FS_AXIS)
    ax.set_title("Per-Type Quality", fontsize=S.FS_TITLE)
    S.add_panel_label(ax, "b")

    # (c) Best/worst genes
    ax = axes[2]
    n_show = 10
    sorted_idx = np.argsort(recon_per_gene)
    best_idx = sorted_idx[:n_show]
    worst_idx = sorted_idx[-n_show:][::-1]
    combined_idx = np.concatenate([worst_idx, best_idx])

    if gene_names is not None:
        gene_labels = gene_names[combined_idx]
    else:
        gene_labels = [f"g{i}" for i in combined_idx]

    gene_vals = recon_per_gene[combined_idx]
    y_pos = np.arange(len(combined_idx))
    colors = ["#D55E00"] * n_show + ["#009E73"] * n_show
    ax.barh(y_pos, gene_vals, color=colors, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gene_labels, fontsize=S.FS_SMALL - 1)
    ax.set_xlabel("Gene MSE", fontsize=S.FS_AXIS)
    ax.set_title("Worst / Best Genes", fontsize=S.FS_TITLE)
    ax.axhline(n_show - 0.5, color="gray", linestyle="--", linewidth=0.5)
    S.add_panel_label(ax, "c")

    if dataset_name:
        fig.suptitle(dataset_name, fontsize=S.FS_TITLE + 1, y=1.02)

    return fig


# ============================================================================
# Figure 10: Hyperbolic Hierarchy Distance Matrix
# ============================================================================

def fig_hierarchy_distances(
    dist_matrix: np.ndarray,
    label_names: List[str],
    lorentz_norms: np.ndarray,
    labels: np.ndarray,
    dataset_name: str = "",
) -> plt.Figure:
    """2-panel: (a) hyperbolic distance heatmap between types,
    (b) hierarchy ordering (type means sorted by norm)."""

    S.apply_style()
    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, S.FIG_WIDTH_IN * 0.42))
    axes = S.row_of_axes(fig, 2, [0.06, 0.18, 0.90, 0.72], gap=0.10,
                          widths=[1.2, 1.0])

    # (a) Distance heatmap
    ax = axes[0]
    n = len(label_names)
    im = ax.imshow(dist_matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, fontsize=S.FS_SMALL, rotation=45, ha="right")
    ax.set_yticklabels(label_names, fontsize=S.FS_SMALL)
    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = dist_matrix[i, j]
            if val > 0:
                text_color = "white" if val > dist_matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=S.FS_SMALL - 1, color=text_color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Hyp. distance")
    ax.set_title("Hyperbolic Distance Matrix", fontsize=S.FS_TITLE)
    S.add_panel_label(ax, "a")

    # (b) Hierarchy ordering
    ax = axes[1]
    unique_labels = sorted(np.unique(labels).tolist())
    palette, _ = _cluster_palette(labels)
    type_means = [(lbl, lorentz_norms[labels == lbl].mean()) for lbl in unique_labels]
    type_means.sort(key=lambda x: x[1])

    y_pos = np.arange(len(type_means))
    bar_colors = [palette[t[0]] for t in type_means]
    norms = [t[1] for t in type_means]
    names = [t[0] for t in type_means]

    ax.barh(y_pos, norms, color=bar_colors, alpha=0.8, height=0.7,
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=S.FS_SMALL)
    ax.set_xlabel("Mean Lorentz norm", fontsize=S.FS_AXIS)
    ax.set_title("Hierarchy Ordering", fontsize=S.FS_TITLE)
    # Arrow annotation: root → leaf
    ax.annotate("root", xy=(norms[0], 0), fontsize=S.FS_SMALL - 1,
                color="gray", ha="right", va="center",
                xytext=(-5, 0), textcoords="offset points")
    ax.annotate("leaf", xy=(norms[-1], len(norms) - 1), fontsize=S.FS_SMALL - 1,
                color="gray", ha="left", va="center",
                xytext=(5, 0), textcoords="offset points")
    S.add_panel_label(ax, "b")

    if dataset_name:
        fig.suptitle(dataset_name, fontsize=S.FS_TITLE + 1, y=1.02)

    return fig


# ============================================================================
# Figure 11: Marker Gene Recovery
# ============================================================================

def fig_marker_recovery(
    marker_overlap: Dict[str, Dict],
    labels: np.ndarray,
    dataset_name: str = "",
) -> plt.Figure:
    """2-panel: (a) overlap fraction per cluster, (b) best matching dimension."""

    S.apply_style()
    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, S.FIG_WIDTH_IN * 0.40))
    axes = S.row_of_axes(fig, 2, [0.08, 0.20, 0.88, 0.68], gap=0.10)
    palette, _ = _cluster_palette(labels)

    clusters = sorted(marker_overlap.keys())
    if not clusters:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.text(0.5, 0.5, "No marker overlap data",
                ha="center", va="center", fontsize=S.FS_AXIS, transform=ax.transAxes)
        ax.axis("off")
        return fig

    # (a) Overlap fraction
    ax = axes[0]
    y_pos = np.arange(len(clusters))
    fracs = [marker_overlap[c]["overlap_fraction"] for c in clusters]
    bar_colors = [palette.get(c, "#999999") for c in clusters]
    ax.barh(y_pos, fracs, color=bar_colors, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clusters, fontsize=S.FS_SMALL)
    ax.set_xlabel("Marker overlap fraction", fontsize=S.FS_AXIS)
    ax.set_title("DE Marker Recovery", fontsize=S.FS_TITLE)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    S.add_panel_label(ax, "a")

    # (b) Best matching dimension
    ax = axes[1]
    best_dims = [marker_overlap[c]["best_matching_dim"] for c in clusters]
    best_overlaps = [marker_overlap[c]["best_dim_overlap"] for c in clusters]
    ax.barh(y_pos, best_overlaps, color=bar_colors, alpha=0.8, height=0.7)
    for i, d in enumerate(best_dims):
        ax.text(best_overlaps[i] + 0.3, i, f"z{d}", fontsize=S.FS_SMALL - 1,
                va="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clusters, fontsize=S.FS_SMALL)
    ax.set_xlabel("Overlap count", fontsize=S.FS_AXIS)
    ax.set_title("Best Latent Dimension", fontsize=S.FS_TITLE)
    ax.invert_yaxis()
    S.add_panel_label(ax, "b")

    if dataset_name:
        fig.suptitle(dataset_name, fontsize=S.FS_TITLE + 1, y=1.02)

    return fig


# ============================================================================
# Figure 12: Cross-Dataset Biovalidation Summary
# ============================================================================

def fig_biovalidation_summary(
    dataset_names: List[str],
    stemness_corrs: List[float],
    mean_marker_overlap: List[float],
    hierarchy_scores: List[float],
    mean_recon_mse: List[float],
    homophily_scores: List[float],
    mean_lorentz_norms: List[float],
) -> plt.Figure:
    """6-panel cross-dataset biovalidation summary."""

    S.apply_style()
    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, S.FIG_HEIGHT_IN * 0.65))
    axes = S.grid_of_axes(fig, 3, 2, [0.08, 0.06, 0.88, 0.88],
                          hgap=0.10, wgap=0.12)
    n = len(dataset_names)
    x = np.arange(n)
    short = [d[:10] for d in dataset_names]

    panels = [
        (axes[0][0], homophily_scores, "GAT Homophily", "#009E73", (0, 1)),
        (axes[0][1], stemness_corrs, "Stemness Corr.", "#0072B2", (-1, 1)),
        (axes[1][0], mean_marker_overlap, "Marker Recovery", "#E69F00", (0, 1)),
        (axes[1][1], hierarchy_scores, "Hierarchy Score", "#CC79A7", None),
        (axes[2][0], mean_lorentz_norms, "Mean Hyp. Radius", "#56B4E9", None),
        (axes[2][1], mean_recon_mse, "Recon. MSE", "#D55E00", None),
    ]

    panel_labels = "abcdef"
    for i, (ax, vals, title, color, ylim) in enumerate(panels):
        ax.bar(x, vals, color=color, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=S.FS_SMALL - 1)
        ax.set_title(title, fontsize=S.FS_TITLE)
        if ylim:
            ax.set_ylim(*ylim)
        S.add_panel_label(ax, panel_labels[i])

    fig.suptitle("Biovalidation Summary", fontsize=S.FS_TITLE + 2, y=1.02)
    return fig


# ============================================================================
# Master generation function
# ============================================================================

def generate_interpretation_figures(
    result,  # InterpretationResult
    output_dir: str | Path,
    umap_coords: Optional[np.ndarray] = None,
) -> List[str]:
    """Generate all interpretation + biovalidation figures for a dataset.

    Parameters
    ----------
    result : InterpretationResult
        Output from run_interpretation() + run_biovalidation().
    output_dir : str or Path
        Directory to save figures.
    umap_coords : ndarray, optional
        Pre-computed UMAP coordinates for overlay plots.

    Returns
    -------
    List of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    ds = result.dataset_name

    # --- Model Interpretation Figures ---

    # Fig 1: Bottleneck analysis
    if result.le is not None:
        fig = fig_bottleneck_analysis(
            result.le, result.q_m, result.q_s,
            result.ib_retention, result.labels, ds,
        )
        paths = S.save_figure(fig, output_dir / f"fig_bottleneck_{ds}.png")
        saved.extend(paths)
        plt.close(fig)

    # Fig 2: Hyperbolic geometry
    if result.poincare_coords is not None:
        fig = fig_hyperbolic_geometry(
            result.poincare_coords, result.lorentz_norms,
            result.labels, ds, umap_coords=umap_coords,
        )
        paths = S.save_figure(fig, output_dir / f"fig_hyperbolic_{ds}.png")
        saved.extend(paths)
        plt.close(fig)

    # Fig 3: GAT attention
    if result.attn_type_matrix is not None and result.attention_weights:
        last_attn = result.attention_weights[-1]
        # Use expanded edge_index (with self-loops) that matches attention weights
        attn_ei = result.attn_edge_index if result.attn_edge_index is not None else result.edge_index
        fig = fig_attention_analysis(
            result.attn_type_matrix, result.label_names,
            result.attn_homophily, attn_ei,
            last_attn, result.labels, ds,
        )
        paths = S.save_figure(fig, output_dir / f"fig_attention_{ds}.png")
        saved.extend(paths)
        plt.close(fig)

    # Fig 4: Gene attribution
    if result.gene_scores is not None and result.gene_names is not None:
        fig = fig_gene_attribution(
            result.gene_scores, result.gene_names,
            top_k=15, dataset_name=ds,
        )
        paths = S.save_figure(fig, output_dir / f"fig_genes_{ds}.png")
        saved.extend(paths)
        plt.close(fig)

    # --- Biovalidation Figures ---

    # Fig 6: Stemness–hierarchy
    if result.stemness_scores is not None and result.stemness_norm_corr is not None:
        fig = fig_stemness_hierarchy(
            result.stemness_scores, result.lorentz_norms,
            result.labels, result.stemness_norm_corr,
            result.stemness_norm_pval, ds, umap_coords=umap_coords,
        )
        paths = S.save_figure(fig, output_dir / f"fig_stemness_{ds}.png")
        saved.extend(paths)
        plt.close(fig)

    # Fig 7: Enrichment summary
    if result.enrichment_results:
        fig = fig_enrichment_summary(result.enrichment_results, ds)
        paths = S.save_figure(fig, output_dir / f"fig_enrichment_{ds}.png")
        saved.extend(paths)
        plt.close(fig)

    # Fig 8: Latent traversal
    if result.traversal_responses and result.gene_names is not None:
        fig = fig_latent_traversal(
            result.traversal_responses, result.gene_names,
            top_k=8, dataset_name=ds,
        )
        paths = S.save_figure(fig, output_dir / f"fig_traversal_{ds}.png")
        saved.extend(paths)
        plt.close(fig)

    # Fig 9: Reconstruction quality
    if result.recon_per_cell is not None:
        fig = fig_reconstruction_quality(
            result.recon_per_cell, result.recon_per_gene,
            result.recon_per_type, result.labels,
            result.gene_names, ds,
        )
        paths = S.save_figure(fig, output_dir / f"fig_recon_{ds}.png")
        saved.extend(paths)
        plt.close(fig)

    # Fig 10: Hierarchy distances
    if result.hyp_dist_matrix is not None:
        fig = fig_hierarchy_distances(
            result.hyp_dist_matrix, result.hyp_dist_labels,
            result.lorentz_norms, result.labels, ds,
        )
        paths = S.save_figure(fig, output_dir / f"fig_hierarchy_{ds}.png")
        saved.extend(paths)
        plt.close(fig)

    # Fig 11: Marker recovery
    if result.marker_overlap:
        fig = fig_marker_recovery(
            result.marker_overlap, result.labels, ds,
        )
        paths = S.save_figure(fig, output_dir / f"fig_markers_{ds}.png")
        saved.extend(paths)
        plt.close(fig)

    return saved


# ============================================================================
# Themed cross-dataset figures (grouped by interpretation theme)
# ============================================================================

import logging as _logging

_themed_logger = _logging.getLogger(__name__)


def _load_all_arrays(
    arrays_dir: str | Path,
    dataset_order: Optional[List[str]] = None,
) -> "OrderedDict[str, dict]":
    """Load all per-cell .npz arrays for themed cross-dataset figures.

    Returns OrderedDict mapping dataset_name -> dict of numpy arrays,
    ordered by *dataset_order* (or alphabetically if not given).
    """
    from collections import OrderedDict

    arrays_dir = Path(arrays_dir)
    available = {p.stem: p for p in sorted(arrays_dir.glob("*.npz"))}

    if dataset_order is None:
        dataset_order = sorted(available.keys())

    data: OrderedDict[str, dict] = OrderedDict()
    for ds in dataset_order:
        if ds in available:
            data[ds] = dict(np.load(available[ds], allow_pickle=True))
        else:
            _themed_logger.warning("No .npz for dataset %s, skipping", ds)
    return data


# Short display names for panel titles (strip GSE prefix noise)
_DS_SHORT = {
    "GSE98638_TcellLiverHmCancer": "TcellLiver",
    "GSE117988_MCCTumorCancer": "MCCTumor",
    "GSE155109_bcECHmCancer": "bcEC",
    "GSE149655_CAHmCancer": "CA",
    "GSE283205_hepatoblastomaCancer": "Hepatobl.",
    "GSE168181_BreastHmCancer": "Breast",
    "endo": "Endo",
    "GSE142653pitHmDev": "Pituitary",
    "setty": "Setty",
    "hESC_GSE144024": "hESC",
    "GSE130148_LungHmDev": "Lung",
    "dentate": "Dentate",
}


def _short_ds(name: str) -> str:
    return _DS_SHORT.get(name, name[:12])


def _section_header(fig, x, y, label, title, fontsize=None):
    """Place a section header with bold panel label and normal title text."""
    fs = fontsize or S.FS_TITLE
    fig.text(x, y, f"({label})",
             fontsize=fs, fontweight="bold", va="bottom", ha="left",
             path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    fig.text(x + 0.025, y, f"  {title}",
             fontsize=fs, fontweight="normal", va="bottom", ha="left")


def _themed_dual_grid(
    all_data: "OrderedDict[str, dict]",
    fill_fn_a,
    fill_fn_b,
    title_a: str,
    title_b: str,
    *,
    gaps_a: Optional[Tuple[float, float]] = None,
    gaps_b: Optional[Tuple[float, float]] = None,
    spines_b: bool = True,
) -> plt.Figure:
    """Helper: two vertically-stacked 3x4 grids on a single 17x21 cm page.

    *fill_fn_a(ax, ds_name, data)* and *fill_fn_b(ax, ds_name, data)* each
    populate one cell.  *gaps_a/gaps_b = (hgap, wgap)* override cell gaps.
    *spines_b=False* removes border spines on grid (b) panels.
    """
    S.apply_style()
    n = len(all_data)
    nrows, ncols = 3, 4

    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, S.FIG_HEIGHT_IN))

    margin_l = 0.03
    grid_w   = 0.93
    grid_h   = 0.42
    wgap_a   = gaps_a[1] if gaps_a else 0.04
    hgap_a   = gaps_a[0] if gaps_a else 0.04
    wgap_b   = gaps_b[1] if gaps_b else wgap_a
    hgap_b   = gaps_b[0] if gaps_b else hgap_a

    top_a    = 0.93
    bot_a    = top_a - grid_h
    top_b    = bot_a - 0.05
    bot_b    = top_b - grid_h

    axes_a = S.grid_of_axes(fig, nrows, ncols,
                            [margin_l, bot_a, grid_w, grid_h],
                            hgap=hgap_a, wgap=wgap_a)
    axes_b = S.grid_of_axes(fig, nrows, ncols,
                            [margin_l, bot_b, grid_w, grid_h],
                            hgap=hgap_b, wgap=wgap_b)

    for idx, (ds_name, data) in enumerate(all_data.items()):
        r, c = idx // ncols, idx % ncols
        fill_fn_a(axes_a[r][c], ds_name, data)
        fill_fn_b(axes_b[r][c], ds_name, data)
        if not spines_b:
            for spine in axes_b[r][c].spines.values():
                spine.set_visible(False)

    for idx in range(n, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes_a[r][c].set_visible(False)
        axes_b[r][c].set_visible(False)

    _section_header(fig, margin_l, top_a + 0.02, "a", title_a)
    _section_header(fig, margin_l, top_b + 0.02, "b", title_b)

    return fig


def _themed_single_grid(
    all_data: "OrderedDict[str, dict]",
    fill_fn,
    suptitle: str,
) -> plt.Figure:
    """Helper: one 3×4 grid on a 17×21 cm page with larger fonts.

    Panel labels are prepended to each subplot title so they never
    overlap with the title text.

    Vertical budget: 3 % suptitle | grid 88 % (hgap=7 % between rows
    to accommodate panel titles) | 5 % bottom.
    """
    S.apply_style()
    n = len(all_data)
    nrows, ncols = 3, 4

    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, S.FIG_HEIGHT_IN))

    grid_rect = [0.03, 0.05, 0.93, 0.87]
    axes = S.grid_of_axes(fig, nrows, ncols, grid_rect,
                          hgap=0.07, wgap=0.05)
    panel_labels = "abcdefghijkl"

    for idx, (ds_name, data) in enumerate(all_data.items()):
        r, c = idx // ncols, idx % ncols
        ax = axes[r][c]
        fill_fn(ax, ds_name, data, idx)
        # Bold panel label via add_panel_label; keep title text normal
        if idx < len(panel_labels):
            S.add_panel_label(ax, panel_labels[idx], fontsize=S.FS_AXIS + 1)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(suptitle, fontsize=S.FS_TITLE + 2, fontweight="normal",
                 y=0.97)
    return fig


# ------------------------------------------------------------------
# Figure T1: Combined UMAP + Poincaré (dual 3×4 grid)
# ------------------------------------------------------------------

def fig_themed_embedding_poincare(
    all_data: "OrderedDict[str, dict]",
) -> plt.Figure:
    """Combined figure: (a) UMAP embedding + (b) Poincaré disk, 3×4 each.

    Fits a single 17×21 cm page (compact layout with minimal row gap).
    """
    from sklearn.decomposition import PCA

    theta = np.linspace(0, 2 * np.pi, 200)

    def _fill_umap(ax, ds_name, data):
        labels = data["labels"]
        if labels.dtype.kind == "O":
            labels = labels.astype(str)
        _, colors = _cluster_palette(labels)
        umap = data["umap_coords"]
        ax.scatter(umap[:, 0], umap[:, 1], c=colors, s=2, alpha=0.6,
                   rasterized=True)
        ax.set_title(_short_ds(ds_name), fontsize=S.FS_AXIS, pad=2)
        ax.set_xticks([]); ax.set_yticks([])

    def _fill_poincare(ax, ds_name, data):
        labels = data["labels"]
        if labels.dtype.kind == "O":
            labels = labels.astype(str)
        _, colors = _cluster_palette(labels)
        poincare = data["poincare_coords"]
        if poincare.shape[1] > 2:
            coords_2d = PCA(n_components=2, random_state=42).fit_transform(
                poincare)
        else:
            coords_2d = poincare
        ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.5, alpha=0.3)
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=colors, s=2,
                   alpha=0.6, rasterized=True)
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.set_title(_short_ds(ds_name), fontsize=S.FS_AXIS, pad=2)
        ax.set_xticks([]); ax.set_yticks([])

    return _themed_dual_grid(
        all_data, _fill_umap, _fill_poincare,
        "UMAP Embedding by Cluster",
        "Poincaré Disk Projection by Cluster",
        gaps_b=(0.02, 0.02),                 # tighter for equal-aspect disks
        spines_b=False,
    )


# ------------------------------------------------------------------
# Figure T2: Bottleneck — IB scatter + Most-Active Latent Dim (dual)
# ------------------------------------------------------------------

def fig_themed_bottleneck(
    all_data: "OrderedDict[str, dict]",
) -> plt.Figure:
    """(a) IB 2-D scatter by cluster + (b) UMAP by most active latent dim."""

    def _fill_ib(ax, ds_name, data):
        le = data["le"]
        labels = data["labels"]
        if labels.dtype.kind == "O":
            labels = labels.astype(str)
        _, colors = _cluster_palette(labels)
        if le.shape[1] >= 2:
            ax.scatter(le[:, 0], le[:, 1], c=colors, s=2, alpha=0.6,
                       rasterized=True)
        else:
            ax.hist(le[:, 0], bins=40, color="#0072B2", alpha=0.7)
        ax.set_title(_short_ds(ds_name), fontsize=S.FS_AXIS, pad=2)
        ax.set_xticks([]); ax.set_yticks([])

    def _fill_latent(ax, ds_name, data):
        umap = data["umap_coords"]
        q_z = data["q_z"]
        dim_var = data["dim_variance"]
        d = int(np.argsort(dim_var)[-1])     # most active dim
        sc = ax.scatter(umap[:, 0], umap[:, 1], c=q_z[:, d],
                        cmap=_NORM_CMAP, s=2, alpha=0.6, rasterized=True)
        ax.set_title(f"{_short_ds(ds_name)} (z{d})",
                     fontsize=S.FS_AXIS, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
        cb.ax.tick_params(labelsize=5)

    return _themed_dual_grid(
        all_data, _fill_ib, _fill_latent,
        "Information Bottleneck 2-D Scatter",
        "Most Active Latent Dimension on UMAP",
        gaps_a=(0.02, 0.02),
        gaps_b=(0.02, 0.02),
    )


# ------------------------------------------------------------------
# Figure T3: Hyperbolic depth — UMAP by norm + Poincaré by norm (dual)
# ------------------------------------------------------------------

def fig_themed_hyperbolic(
    all_data: "OrderedDict[str, dict]",
) -> plt.Figure:
    """(a) UMAP coloured by Lorentz norm + (b) Poincaré coloured by norm."""
    from sklearn.decomposition import PCA

    theta = np.linspace(0, 2 * np.pi, 200)

    def _fill_umap_norm(ax, ds_name, data):
        umap = data["umap_coords"]
        norms = data["lorentz_norms"]
        sc = ax.scatter(umap[:, 0], umap[:, 1], c=norms, cmap=_NORM_CMAP,
                        s=2, alpha=0.6, rasterized=True)
        ax.set_title(_short_ds(ds_name), fontsize=S.FS_AXIS, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
        cb.ax.tick_params(labelsize=5)

    def _fill_poincare_norm(ax, ds_name, data):
        poincare = data["poincare_coords"]
        norms = data["lorentz_norms"]
        if poincare.shape[1] > 2:
            coords_2d = PCA(n_components=2, random_state=42).fit_transform(
                poincare)
        else:
            coords_2d = poincare
        ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.5, alpha=0.3)
        sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=norms,
                        cmap=_NORM_CMAP, s=2, alpha=0.6, rasterized=True)
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.set_title(_short_ds(ds_name), fontsize=S.FS_AXIS, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
        cb.ax.tick_params(labelsize=5)

    return _themed_dual_grid(
        all_data, _fill_umap_norm, _fill_poincare_norm,
        "UMAP Coloured by Lorentz Norm",
        "Poincaré Disk Coloured by Lorentz Norm",
        gaps_b=(0.02, 0.02),
        spines_b=False,
    )


# ------------------------------------------------------------------
# Figure T4: Summary + Stemness  (bar charts  +  UMAP 3×4 grid)
# ------------------------------------------------------------------

def _load_bioval_summaries(tables_dir: Path, ds_order) -> dict:
    """Load per-dataset biovalidation JSONs into metric arrays."""
    import json

    metrics = {
        "homophily": [], "lorentz_norm": [], "ib_retention": [],
        "active_dims": [], "stemness_corr": [], "marker_overlap": [],
        "hierarchy_score": [], "recon_mse": [],
    }
    names = []
    for ds in ds_order:
        jpath = tables_dir / f"interp_{ds}_bioval.json"
        if not jpath.exists():
            continue
        d = json.loads(jpath.read_text())
        names.append(ds)
        metrics["homophily"].append(d.get("attn_homophily", 0))
        metrics["lorentz_norm"].append(d.get("mean_lorentz_norm", 0))
        metrics["ib_retention"].append(d.get("mean_ib_retention", 0))
        metrics["active_dims"].append(d.get("n_active_dims", 0))
        metrics["stemness_corr"].append(d.get("stemness_corr", 0))
        metrics["marker_overlap"].append(d.get("mean_marker_overlap", 0))
        metrics["hierarchy_score"].append(d.get("hierarchy_score", 0))
        metrics["recon_mse"].append(d.get("mean_recon_mse", 0))
    return names, metrics


def fig_themed_summary(
    all_data: "OrderedDict[str, dict]",
    tables_dir: Path,
) -> plt.Figure:
    """Composed figure: (a) cross-dataset bar-chart summary,
    (b) stemness projection 3×4 grid.

    Every text element (titles, x-tick labels, section headers) has an
    explicit vertical allocation so nothing is ever masked.

    Vertical budget (normalised, top → bottom):
      0.97  section (a) title
      0.83–0.95  bar row 1 axes  (h=0.12)
      0.77–0.83  row 1 xtick zone  (0.06)
      0.76  row gap
      0.60–0.72  bar row 2 axes  (h=0.12)
      0.54–0.60  row 2 xtick zone  (0.06)
      0.50  section (b) title
      0.06–0.47  stemness grid  (h=0.41)
      0.06  bottom margin
    """
    S.apply_style()
    ds_order = list(all_data.keys())
    ds_names, mets = _load_bioval_summaries(tables_dir, ds_order)
    short = [_short_ds(n) for n in ds_names]
    n_ds = len(ds_names)

    fig_w = S.FIG_WIDTH_IN
    fig_h = S.FIG_HEIGHT_IN
    fig = plt.figure(figsize=(fig_w, fig_h))

    # === Section (a): 2×4 bar chart grid ================================
    _row1 = [
        ("GAT Homophily",      mets["homophily"],      "#009E73", (0, 1.05)),
        ("Lorentz Norm",       mets["lorentz_norm"],    "#0072B2", None),
        ("IB Retention (MSE)", mets["ib_retention"],    "#D55E00", None),
        ("Active Dims",        mets["active_dims"],     "#CC79A7", None),
    ]
    _row2 = [
        ("Stemness Corr.",   mets["stemness_corr"],   "#0072B2", (-1, 1)),
        ("Marker Recovery",  mets["marker_overlap"],  "#E69F00", (0, 1.05)),
        ("Hierarchy Score",  mets["hierarchy_score"], "#CC79A7", None),
        ("Recon. MSE",       mets["recon_mse"],       "#D55E00", None),
    ]

    # Panel geometry — explicit zones for axes vs text
    ml   = 0.07                              # y-label room
    pw   = 0.195                             # bar panel width
    pg   = 0.035                             # horizontal gap
    ax_h = 0.12                              # bar axes height

    row1_bot = 0.82                          # row 1 axes bottom
    row2_bot = 0.64                          # row 2 axes bottom

    def _make_bar_row(panels, axes_bot):
        for ci, (title, vals, color, ylim) in enumerate(panels):
            left = ml + ci * (pw + pg)
            ax = fig.add_axes([left, axes_bot, pw, ax_h])
            x = np.arange(n_ds)
            ax.bar(x, vals, color=color, alpha=0.85, width=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(short, rotation=50, ha="right",
                               fontsize=6)
            ax.tick_params(axis="y", labelsize=6)
            ax.tick_params(axis="x", pad=1)
            ax.set_title(title, fontsize=S.FS_SMALL + 1, pad=3)
            if ylim:
                ax.set_ylim(*ylim)

    _make_bar_row(_row1, row1_bot)
    _make_bar_row(_row2, row2_bot)

    _section_header(fig, 0.03, 0.985, "a", "Cross-Dataset Summary")

    # === Section (b): Stemness UMAP 3×4 grid ============================
    nrows, ncols = 3, 4
    stem_bot = 0.05
    stem_h   = 0.48

    axes_stem = S.grid_of_axes(fig, nrows, ncols,
                               [0.03, stem_bot, 0.93, stem_h],
                               hgap=0.06, wgap=0.05)

    for idx, (ds_name, data) in enumerate(all_data.items()):
        r, c = idx // ncols, idx % ncols
        ax = axes_stem[r][c]
        umap = data["umap_coords"]
        stem = data["stemness_scores"]
        sc = ax.scatter(umap[:, 0], umap[:, 1], c=stem, cmap="coolwarm",
                        s=2, alpha=0.6, rasterized=True)
        ax.set_title(_short_ds(ds_name), fontsize=S.FS_AXIS, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
        cb.ax.tick_params(labelsize=5)

    n = len(all_data)
    for idx in range(n, nrows * ncols):
        axes_stem[idx // ncols][idx % ncols].set_visible(False)

    _section_header(fig, 0.03, stem_bot + stem_h + 0.02, "b",
                    "Stemness Projection on UMAP")

    return fig


# ------------------------------------------------------------------
# Figure T5: Gene expression — top 3 genes composed (triple 3×4 grid)
# ------------------------------------------------------------------

def fig_themed_gene_expression(
    all_data: "OrderedDict[str, dict]",
) -> plt.Figure:
    """Composed figure with three 3×4 grids: (a) top gene, (b) 2nd gene,
    (c) 3rd gene.  Each cell = UMAP coloured by gene expression.

    Vertical budget (normalised):
      top 2 % | title_a 2 % | grid_a 28 % | gap+title_b 4 % |
      grid_b 28 % | gap+title_c 4 % | grid_c 28 % | bottom 4 %
    """
    S.apply_style()
    n = len(all_data)
    nrows, ncols = 3, 4

    fig = plt.figure(figsize=(S.FIG_WIDTH_IN, S.FIG_HEIGHT_IN))

    margin_l = 0.03
    grid_w   = 0.93
    grid_h   = 0.28
    wgap     = 0.04
    hgap     = 0.03                          # tight within each grid

    # Three stacked grids
    top_a  = 0.94;  bot_a = top_a - grid_h  # 0.66
    top_b  = bot_a - 0.04;  bot_b = top_b - grid_h  # 0.34
    top_c  = bot_b - 0.04;  bot_c = top_c - grid_h  # 0.02

    axes_a = S.grid_of_axes(fig, nrows, ncols,
                            [margin_l, bot_a, grid_w, grid_h],
                            hgap=hgap, wgap=wgap)
    axes_b = S.grid_of_axes(fig, nrows, ncols,
                            [margin_l, bot_b, grid_w, grid_h],
                            hgap=hgap, wgap=wgap)
    axes_c = S.grid_of_axes(fig, nrows, ncols,
                            [margin_l, bot_c, grid_w, grid_h],
                            hgap=hgap, wgap=wgap)

    ordinals = ["Top", "2nd", "3rd"]

    for rank, (axes_g, top_g) in enumerate(
            [(axes_a, top_a), (axes_b, top_b), (axes_c, top_c)]):
        for idx, (ds_name, data) in enumerate(all_data.items()):
            r, c = idx // ncols, idx % ncols
            ax = axes_g[r][c]
            umap = data["umap_coords"]
            expr = data["top_gene_expr"]
            names = data["top_gene_expr_names"]
            if rank >= expr.shape[1]:
                ax.set_visible(False)
                continue
            sc = ax.scatter(umap[:, 0], umap[:, 1], c=expr[:, rank],
                            cmap=_GENE_CMAP, s=2, alpha=0.6, rasterized=True)
            gene_name = str(names[rank])
            ax.set_title(f"{_short_ds(ds_name)}: {gene_name}",
                         fontsize=S.FS_SMALL, pad=1)
            ax.set_xticks([]); ax.set_yticks([])
            cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
            cb.ax.tick_params(labelsize=4)

        for idx in range(n, nrows * ncols):
            axes_g[idx // ncols][idx % ncols].set_visible(False)

        label = chr(ord("a") + rank)
        _section_header(fig, margin_l, top_g + 0.015, label,
                        f"{ordinals[rank]} Attributed Gene",
                        fontsize=S.FS_AXIS)

    return fig


# ------------------------------------------------------------------
# Master themed-figure generator
# ------------------------------------------------------------------

def generate_themed_figures(
    arrays_dir: str | Path,
    output_dir: str | Path,
    dataset_order: Optional[List[str]] = None,
) -> List[str]:
    """Generate all themed cross-dataset interpretation figures.

    Parameters
    ----------
    arrays_dir : path
        Directory containing per-dataset .npz files.
    output_dir : path
        Directory to save themed figures.
    dataset_order : list of str, optional
        Canonical dataset order for grid layout.

    Returns
    -------
    List of saved file paths.
    """
    all_data = _load_all_arrays(arrays_dir, dataset_order)
    if len(all_data) < 2:
        _themed_logger.warning(
            "Need >= 2 datasets for themed figures, found %d", len(all_data))
        return []

    output_dir = Path(output_dir)
    arrays_path = Path(arrays_dir)
    tables_dir = arrays_path.parent / "tables"
    saved = []

    # Remove ALL outdated files that are no longer generated
    import shutil
    _outdated_prefixes = [
        "fig_themed_embedding", "fig_themed_poincare",
        "fig_themed_stemness", "fig_themed_latent_dim",
        "fig_biovalidation_summary", "fig_interpretation_summary",
        "fig_themed_gene1", "fig_themed_gene2", "fig_themed_gene3",
    ]
    for p in list(output_dir.glob("*")):
        if p.is_dir():
            # Per-dataset sub-directories
            shutil.rmtree(p)
            _themed_logger.info("Removed per-dataset dir %s", p)
        elif any(p.stem.startswith(pf) for pf in _outdated_prefixes):
            p.unlink()
            _themed_logger.info("Removed outdated %s", p)

    # T1: Combined embedding + Poincaré (dual grid)
    fig = fig_themed_embedding_poincare(all_data)
    saved.extend(S.save_figure(
        fig, output_dir / "fig_themed_embedding_poincare.png"))
    plt.close(fig)

    # T2: Bottleneck — IB scatter + most active latent dim (dual grid)
    fig = fig_themed_bottleneck(all_data)
    saved.extend(S.save_figure(
        fig, output_dir / "fig_themed_bottleneck.png"))
    plt.close(fig)

    # T3: Hyperbolic depth — UMAP + Poincaré by Lorentz norm (dual grid)
    fig = fig_themed_hyperbolic(all_data)
    saved.extend(S.save_figure(
        fig, output_dir / "fig_themed_hyperbolic.png"))
    plt.close(fig)

    # T4: Summary bars + Stemness grid (composed)
    if tables_dir.exists():
        fig = fig_themed_summary(all_data, tables_dir)
        saved.extend(S.save_figure(
            fig, output_dir / "fig_themed_summary.png"))
        plt.close(fig)

    # T5: Gene expression (top 3 genes merged into one figure)
    fig = fig_themed_gene_expression(all_data)
    saved.extend(S.save_figure(
        fig, output_dir / "fig_themed_genes.png"))
    plt.close(fig)

    _themed_logger.info("Themed figures: saved %d files to %s",
                        len(saved), output_dir)
    return saved
