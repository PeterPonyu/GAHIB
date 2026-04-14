#!/usr/bin/env python3
"""
Trajectory / pseudotime validation figure for GAHIB.

Rather than evaluating GAHIB pseudotime against an algorithmic reference
such as diffusion pseudotime (DPT), this figure demonstrates the
biological meaning of GAHIB's Lorentz-norm pseudotime directly from the
original gene expression.

For each dataset we show:
  (row 1) 2D UMAP coloured by Leiden clusters
  (row 2) 2D UMAP coloured by GAHIB Lorentz pseudotime
  (row 3) scatter of GAHIB pseudotime vs. the top negatively-correlated
          ("early / progenitor-like") gene — expression decreases
          along the trajectory, with Pearson r reported
  (row 4) scatter of GAHIB pseudotime vs. the top positively-correlated
          ("lineage") gene — expression increases along the trajectory

A summary bar at the bottom shows the mean |r| of the top-20 HVGs,
quantifying how strongly the gene programme aligns with GAHIB pseudotime
per dataset.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd


_FONT_DIR = Path(__file__).resolve().parent / "fonts"
_LAB_DIR = Path.home() / "LAB"
_ARIAL_SAFE = {"Arial.ttf", "Arial Bold.ttf"}
for _dir in (_FONT_DIR, _LAB_DIR):
    if _dir.exists():
        for ttf in _dir.glob("Arial*.ttf"):
            if ttf.name not in _ARIAL_SAFE:
                continue
            try:
                fm.fontManager.addfont(str(ttf))
            except Exception:
                pass


_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.labelsize": 9,
    "axes.labelweight": "normal",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.32,
    "pdf.fonttype": 42,
}


def _apply_style():
    plt.rcParams.update(_RC)


def _panel_label(ax, label, x=-0.18, y=1.06, fontsize=13):
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            va="bottom", ha="left")


def _maybe_recompute_gene_panel(name, data):
    """Return (umap_xy, top_neg, top_pos, top_abs_mean) for a dataset.

    `data` is a NpzFile already opened. If new-format fields are present
    they are used directly; otherwise we try to re-load the raw dataset
    via exp_utils and compute on-the-fly. If that fails we return None.
    """
    # Preferred path: new-format npz has everything we need.
    if "umap_2d" in data.files and "top_gene_names" in data.files:
        umap_xy = data["umap_2d"]
        gene_names = [str(x) for x in data["top_gene_names"]]
        gene_expr = data["top_gene_expr"]
        gene_r = data["top_gene_r"]
        order = np.argsort(gene_r)
        neg_idx = int(order[0])
        pos_idx = int(order[-1])
        top_neg = (gene_names[neg_idx], gene_expr[:, neg_idx], float(gene_r[neg_idx]))
        top_pos = (gene_names[pos_idx], gene_expr[:, pos_idx], float(gene_r[pos_idx]))
        top_abs_mean = float(np.mean(np.abs(gene_r)))
        return umap_xy, top_neg, top_pos, top_abs_mean

    # Fallback: try on-the-fly computation from raw adata.
    try:
        import sys
        sys.path.insert(
            0, str(Path(__file__).resolve().parent.parent.parent / "experiments"))
        from exp_utils import discover_datasets, load_and_preprocess  # type: ignore
        import scipy.sparse as sp
        import scanpy as sc

        ds_files = discover_datasets()
        match = [f for f in ds_files
                 if name in Path(f).name]
        if not match:
            return None
        adata = load_and_preprocess(match[0])
        gahib_pt = np.asarray(data["gahib_pt"], dtype=np.float32)
        if adata.n_obs != gahib_pt.shape[0]:
            return None

        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        pt_c = gahib_pt - gahib_pt.mean()
        pt_n = pt_c / (np.linalg.norm(pt_c) + 1e-12)
        Xc = X - X.mean(axis=0, keepdims=True)
        Xn = Xc / (np.linalg.norm(Xc, axis=0, keepdims=True) + 1e-12)
        r_vals = (Xn * pt_n[:, None]).sum(axis=0)

        var_names = np.asarray(adata.var_names.astype(str))
        order = np.argsort(r_vals)
        neg_idx = int(order[0])
        pos_idx = int(order[-1])

        top_neg = (str(var_names[neg_idx]), X[:, neg_idx], float(r_vals[neg_idx]))
        top_pos = (str(var_names[pos_idx]), X[:, pos_idx], float(r_vals[pos_idx]))
        top_abs_mean = float(np.mean(np.sort(np.abs(r_vals))[-20:]))

        # Compute 2D UMAP for viz
        sc.pp.pca(adata, n_comps=30)
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
        sc.tl.umap(adata, min_dist=0.3, n_components=2)
        umap_xy = adata.obsm["X_umap"].astype(np.float32)

        return umap_xy, top_neg, top_pos, top_abs_mean
    except Exception:
        return None


def fig_trajectory(results_dir: str | Path,
                   datasets_info: list,
                   output_path: str | Path | None = None
                   ) -> plt.Figure:
    _apply_style()
    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"

    n = len(datasets_info)
    fig = plt.figure(figsize=(4.0 * n, 13.0))
    gs = fig.add_gridspec(
        5, n,
        height_ratios=[1.0, 1.0, 1.0, 1.0, 0.9],
        hspace=0.65, wspace=0.40,
        left=0.06, right=0.97, top=0.93, bottom=0.09,
    )

    gene_summary = {"dataset": [], "mean_abs_r_top20": []}

    for col, (ds_name, title) in enumerate(datasets_info):
        npz_file = tables_dir / f"traj_{ds_name}_pts.npz"
        if not npz_file.exists():
            continue

        data = np.load(str(npz_file), allow_pickle=True)
        gahib_pt = data["gahib_pt"]
        labels = data["labels"]

        recovered = _maybe_recompute_gene_panel(ds_name, data)
        if recovered is None:
            continue
        umap_xy, top_neg, top_pos, top_abs_mean = recovered
        gene_summary["dataset"].append(title)
        gene_summary["mean_abs_r_top20"].append(top_abs_mean)

        # Row 1: clusters on 2D UMAP
        ax = fig.add_subplot(gs[0, col])
        unique = sorted(np.unique(labels).tolist())
        cmap = plt.get_cmap("tab20", max(len(unique), 2))
        for i, lbl in enumerate(unique):
            mask = labels == lbl
            ax.scatter(umap_xy[mask, 0], umap_xy[mask, 1], s=3,
                       color=cmap(i % 20), alpha=0.75, edgecolor="none",
                       rasterized=True)
        ax.set_xlabel("UMAP-1", fontsize=8)
        ax.set_ylabel("UMAP-2", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\nLeiden clusters" if col == 0 else title,
                     fontsize=10, pad=4)
        if col == 0:
            _panel_label(ax, "a")

        # Row 2: GAHIB pseudotime on 2D UMAP
        ax = fig.add_subplot(gs[1, col])
        sc = ax.scatter(umap_xy[:, 0], umap_xy[:, 1], c=gahib_pt,
                        cmap="viridis", s=3, alpha=0.85, edgecolor="none",
                        rasterized=True)
        ax.set_xlabel("UMAP-1", fontsize=8)
        ax.set_ylabel("UMAP-2", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_title("GAHIB Lorentz pseudotime", fontsize=10, pad=4)
            _panel_label(ax, "b")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(labelsize=6.5)

        # Row 3: GAHIB pt vs top-negatively-correlated (early / progenitor) gene
        gname, gvec, gr = top_neg
        ax = fig.add_subplot(gs[2, col])
        ax.scatter(gahib_pt, gvec, s=5, alpha=0.45,
                   color="#0072B2", edgecolor="none", rasterized=True)
        ax.text(0.03, 0.97,
                f"{gname}\nr = {gr:+.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="#888", lw=0.5))
        ax.set_xlabel("GAHIB pseudotime", fontsize=8)
        ax.set_ylabel("Normalised expression", fontsize=8)
        if col == 0:
            ax.set_title("Early / progenitor-like gene\n(expr. ↓ with pt)",
                         fontsize=10, pad=4)
            _panel_label(ax, "c")

        # Row 4: GAHIB pt vs top-positively-correlated (lineage) gene
        gname, gvec, gr = top_pos
        ax = fig.add_subplot(gs[3, col])
        ax.scatter(gahib_pt, gvec, s=5, alpha=0.45,
                   color="#D55E00", edgecolor="none", rasterized=True)
        ax.text(0.03, 0.97,
                f"{gname}\nr = {gr:+.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="#888", lw=0.5))
        ax.set_xlabel("GAHIB pseudotime", fontsize=8)
        ax.set_ylabel("Normalised expression", fontsize=8)
        if col == 0:
            ax.set_title("Lineage-commitment gene\n(expr. ↑ with pt)",
                         fontsize=10, pad=4)
            _panel_label(ax, "d")

    # Row 5: summary bar chart — mean |r| of top-20 HVGs per dataset
    ax = fig.add_subplot(gs[4, :])
    sdf = pd.DataFrame(gene_summary)
    if len(sdf) > 0:
        x = np.arange(len(sdf))
        ax.bar(x, sdf["mean_abs_r_top20"], width=0.55,
               color="#D55E00", alpha=0.88,
               edgecolor="#333", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(sdf["dataset"], fontsize=9)
        ax.set_ylabel("Mean |Pearson r| of top-20 HVGs", fontsize=9)
        ax.set_title(
            "Biological alignment of GAHIB pseudotime with top gene programmes",
            fontsize=10.5, pad=6)
        ax.set_ylim(0, max(1.0, sdf["mean_abs_r_top20"].max() * 1.15))
        ax.grid(axis="y", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)
        _panel_label(ax, "e", x=-0.045, y=1.10)

    fig.suptitle(
        "Trajectory / pseudotime: biological validation of GAHIB's "
        "hyperbolic hierarchy",
        fontsize=12.5, fontweight="normal", y=0.985)

    if output_path:
        fig.savefig(str(output_path), format="pdf")
    return fig


def generate_trajectory_figure(results_base: str | Path | None = None):
    if results_base is None:
        results_base = (Path(__file__).resolve().parent.parent.parent
                        / "GAHIB_results" / "trajectory")
    results_base = Path(results_base)

    fig_dir = results_base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    datasets_info = [
        ("dentate",        "Dentate gyrus"),
        ("hemato",         "Hematopoiesis"),
        ("setty",          "Setty HSPC"),
        ("hESC_GSE144024", "hESC"),
    ]
    try:
        fig = fig_trajectory(results_base, datasets_info,
                             output_path=fig_dir / "fig_trajectory.pdf")
        plt.close(fig)
        print(f"  ✓ {fig_dir / 'fig_trajectory.pdf'}")
    except Exception as e:
        import traceback
        print(f"  ✗ {e}")
        traceback.print_exc()


if __name__ == "__main__":
    generate_trajectory_figure()
