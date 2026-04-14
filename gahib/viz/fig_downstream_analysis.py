#!/usr/bin/env python3
"""
Unified downstream-analysis figure for GAHIB.

Combines two biology-facing validations in a single publication figure:
  (top)    Pseudotime / trajectory validation via gene-expression
           correlations with GAHIB's Lorentz-norm pseudotime.
  (bottom) GO Biological-Process enrichment of GAHIB latent dimensions
           (tissue-specific panels).

Panel labels are bold; all other text uses the regular Arial weight.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd


# ── Font registration (prefers LAB folder, falls back to bundled fonts) ──
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
    "savefig.pad_inches": 0.35,
    "pdf.fonttype": 42,
}


def _apply_style():
    plt.rcParams.update(_RC)


def _row_label_from_axes(fig, axes, label, *, dx=0.015, fontsize=20):
    """Emit a single bold panel label at the vertical centre of the row
    that `axes` belongs to. The label is placed at dx to the left of
    the axes' left edge in figure coordinates so it never overlaps
    subplot content or tick labels.
    """
    ax = axes if not isinstance(axes, (list, tuple)) else axes[0]
    bbox = ax.get_position()
    fig.text(bbox.x0 - dx, bbox.y0 + bbox.height / 2.0, f"({label})",
             fontsize=fontsize, fontweight="bold",
             ha="right", va="center")


def _wrap_term(term: str, width: int = 22, max_lines: int = 3) -> str:
    """Strip the trailing GO:id, normalise GOBP_ prefix, then wrap the
    human-readable text onto multiple lines at word boundaries so long
    pathway names fit inside narrow heatmap y-axes.
    """
    if "(GO:" in term:
        term = term.split("(GO:")[0].strip()
    if term.startswith("GOBP_"):
        term = term[len("GOBP_"):].replace("_", " ").title()
    import textwrap
    lines = textwrap.wrap(term, width=width, break_long_words=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip() + "…"
    return "\n".join(lines)


def _top_k_abs_mean(r_vals: np.ndarray, k: int = 20) -> float:
    return float(np.mean(np.sort(np.abs(r_vals))[-k:]))


def _pearson_vs_genes(pt: np.ndarray, X: np.ndarray) -> np.ndarray:
    pt_c = pt - pt.mean()
    pt_n = pt_c / (np.linalg.norm(pt_c) + 1e-12)
    Xc = X - X.mean(axis=0, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=0, keepdims=True) + 1e-12)
    return (Xn * pt_n[:, None]).sum(axis=0)


def _load_trajectory_panel(tables_dir: Path, ds_name: str):
    """Return a dict with UMAP coords, labels, GAHIB pseudotime, the top
    ±-correlated gene tracks, and baseline mean |r| for PCA / UMAP
    pseudotimes — or None if the dataset can't be reconstructed."""
    npz_file = tables_dir / f"traj_{ds_name}_pts.npz"
    if not npz_file.exists():
        return None
    data = np.load(str(npz_file), allow_pickle=True)
    gahib_pt = np.asarray(data["gahib_pt"], dtype=np.float32)
    pca_pt = np.asarray(data["pca_pt"], dtype=np.float32)
    umap_pt = np.asarray(data["umap_pt"], dtype=np.float32)

    # Try to re-load adata via exp_utils so correlations are computed
    # against the raw gene panel (not just GAHIB's top-20 slice).
    try:
        import sys
        sys.path.insert(
            0, str(Path(__file__).resolve().parent.parent.parent / "experiments"))
        from exp_utils import discover_datasets, load_and_preprocess  # type: ignore
        import scipy.sparse as sp
        import scanpy as sc

        match = [f for f in discover_datasets() if ds_name in Path(f).name]
        if not match or len(gahib_pt) == 0:
            return None
        adata = load_and_preprocess(match[0])
        if adata.n_obs != gahib_pt.shape[0]:
            return None
        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        X = np.asarray(X, dtype=np.float32)

        r_gahib = _pearson_vs_genes(gahib_pt, X)
        r_pca = _pearson_vs_genes(pca_pt, X)
        r_umap = _pearson_vs_genes(umap_pt, X)

        var_names = np.asarray(adata.var_names.astype(str))
        order = np.argsort(r_gahib)
        neg_i, pos_i = int(order[0]), int(order[-1])
        top_neg = (str(var_names[neg_i]), X[:, neg_i], float(r_gahib[neg_i]))
        top_pos = (str(var_names[pos_i]), X[:, pos_i], float(r_gahib[pos_i]))

        sc.pp.pca(adata, n_comps=30)
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
        sc.tl.umap(adata, min_dist=0.3, n_components=2)
        umap_xy = adata.obsm["X_umap"].astype(np.float32)

        return {
            "umap_xy": umap_xy,
            "labels": data["labels"],
            "gahib_pt": gahib_pt,
            "top_neg": top_neg,
            "top_pos": top_pos,
            "abs_mean_gahib": _top_k_abs_mean(r_gahib),
            "abs_mean_pca": _top_k_abs_mean(r_pca),
            "abs_mean_umap": _top_k_abs_mean(r_umap),
        }
    except Exception:
        return None


def fig_downstream_analysis(trajectory_base: str | Path,
                             go_base: str | Path,
                             trajectory_datasets: list,
                             go_datasets: list,
                             output_path: str | Path | None = None,
                             top_k_go: int = 3,
                             ) -> plt.Figure:
    """Build a single downstream-analysis figure.

    Parameters
    ----------
    trajectory_base : path to GAHIB_results/trajectory/
    go_base         : path to GAHIB_results/interpretation/go_enrichment/
    trajectory_datasets : list of (dataset_key, display_title)
    go_datasets         : list of (dataset_key, display_title, tissue_string)
    """
    _apply_style()
    trajectory_base = Path(trajectory_base)
    go_base = Path(go_base)
    tables_dir = trajectory_base / "tables"

    n_traj = len(trajectory_datasets)
    n_go = len(go_datasets)

    fig = plt.figure(figsize=(4.0 * n_traj, 19.0))

    # Layout constants: five trajectory rows (a-e) + one GO row (f) share
    # the page. Labels are placed relative to the first column's axes
    # bbox after layout, so hspace/wspace don't drift them off-centre.
    TOP_T = 0.965
    TOP_B = 0.545
    BOT_T = 0.495
    BOT_B = 0.06
    TOP_HSPACE = 0.60
    top_height_ratios = [1.0, 1.0, 1.0, 1.0, 0.85]

    # Top gridspec (trajectory panels). Left margin reserves room for
    # the row labels without letting them bleed into the plot area.
    top_gs = fig.add_gridspec(
        5, n_traj,
        height_ratios=top_height_ratios,
        hspace=TOP_HSPACE, wspace=0.40,
        left=0.10, right=0.97,
        top=TOP_T, bottom=TOP_B,
    )

    # Collect the leftmost axes of each trajectory row for label
    # placement after everything is drawn.
    row_axes: list[plt.Axes] = [None] * 5

    gene_summary_data = []  # list of (title, abs_mean_gahib, pca, umap)

    for col, (ds_name, title) in enumerate(trajectory_datasets):
        rec = _load_trajectory_panel(tables_dir, ds_name)
        if rec is None:
            continue
        umap_xy = rec["umap_xy"]
        labels = rec["labels"]
        gahib_pt = rec["gahib_pt"]
        top_neg = rec["top_neg"]
        top_pos = rec["top_pos"]
        gene_summary_data.append(
            (title, rec["abs_mean_gahib"], rec["abs_mean_pca"],
             rec["abs_mean_umap"])
        )

        # Row 1: clusters
        ax = fig.add_subplot(top_gs[0, col])
        if col == 0:
            row_axes[0] = ax
        unique = sorted(np.unique(labels).tolist())
        cmap = plt.get_cmap("tab20", max(len(unique), 2))
        for i, lbl in enumerate(unique):
            m = labels == lbl
            ax.scatter(umap_xy[m, 0], umap_xy[m, 1], s=3,
                       color=cmap(i % 20), alpha=0.75, edgecolor="none",
                       rasterized=True)
        ax.set_xlabel("UMAP-1", fontsize=8)
        ax.set_ylabel("UMAP-2", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title} — Leiden clusters" if col == 0 else title,
                     fontsize=10, pad=4)

        # Row 2: GAHIB pt
        ax = fig.add_subplot(top_gs[1, col])
        if col == 0:
            row_axes[1] = ax
        sc = ax.scatter(umap_xy[:, 0], umap_xy[:, 1], c=gahib_pt,
                        cmap="viridis", s=3, alpha=0.85, edgecolor="none",
                        rasterized=True)
        ax.set_xlabel("UMAP-1", fontsize=8)
        ax.set_ylabel("UMAP-2", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_title("GAHIB Lorentz pseudotime", fontsize=10, pad=4)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(labelsize=6.5)

        # Row 3: top negative gene
        gname, gvec, gr = top_neg
        ax = fig.add_subplot(top_gs[2, col])
        if col == 0:
            row_axes[2] = ax
        ax.scatter(gahib_pt, gvec, s=5, alpha=0.45,
                   color="#0072B2", edgecolor="none", rasterized=True)
        ax.text(0.03, 0.97,
                f"{gname}\nr = {gr:+.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="#888", lw=0.5))
        ax.set_xlabel("GAHIB pseudotime", fontsize=8)
        ax.set_ylabel("Norm. expression", fontsize=8)
        if col == 0:
            ax.set_title("Early / progenitor-like gene (expr. ↓ with pt)",
                         fontsize=10, pad=4)

        # Row 4: top positive gene
        gname, gvec, gr = top_pos
        ax = fig.add_subplot(top_gs[3, col])
        if col == 0:
            row_axes[3] = ax
        ax.scatter(gahib_pt, gvec, s=5, alpha=0.45,
                   color="#D55E00", edgecolor="none", rasterized=True)
        ax.text(0.03, 0.97,
                f"{gname}\nr = {gr:+.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="#888", lw=0.5))
        ax.set_xlabel("GAHIB pseudotime", fontsize=8)
        ax.set_ylabel("Norm. expression", fontsize=8)
        if col == 0:
            ax.set_title("Lineage-commitment gene (expr. ↑ with pt)",
                         fontsize=10, pad=4)

    # Row 5 (summary bars spanning all trajectory columns): compare the
    # biological alignment of GAHIB's hyperbolic pseudotime with two
    # baseline pseudotimes (PCA distance, UMAP distance) by the mean |r|
    # of the top-20 HVGs.
    ax = fig.add_subplot(top_gs[4, :])
    row_axes[4] = ax
    if gene_summary_data:
        titles = [t for t, *_ in gene_summary_data]
        g_vals = [g for _, g, _, _ in gene_summary_data]
        p_vals = [p for _, _, p, _ in gene_summary_data]
        u_vals = [u for _, _, _, u in gene_summary_data]
        x = np.arange(len(titles))
        w = 0.26
        ax.bar(x - w, g_vals, w, color="#D55E00", alpha=0.92,
               edgecolor="#333", linewidth=0.5, label="GAHIB Lorentz pt")
        ax.bar(x,     p_vals, w, color="#0072B2", alpha=0.85,
               edgecolor="#333", linewidth=0.5, label="PCA distance")
        ax.bar(x + w, u_vals, w, color="#009E73", alpha=0.85,
               edgecolor="#333", linewidth=0.5, label="UMAP distance")
        ax.set_xticks(x)
        ax.set_xticklabels(titles, fontsize=9)
        ax.set_ylabel("Mean |Pearson r| of top-20 HVGs", fontsize=9)
        ax.set_title(
            "Biological alignment of GAHIB pseudotime vs. PCA / UMAP "
            "baselines", fontsize=10.5, pad=6)
        y_ceiling = max(max(g_vals + p_vals + u_vals) * 1.18, 0.5)
        ax.set_ylim(0, y_ceiling)
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
        ax.grid(axis="y", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)

    # Emit one row-level panel label per row (a-e), vertically centred.
    for letter, ax_row in zip("abcde", row_axes):
        if ax_row is not None:
            _row_label_from_axes(fig, ax_row, letter)

    # ── Bottom region: tissue-specific GO enrichment panels ──
    # Wrapped GO term labels free up horizontal space; tighten wspace
    # so the four heatmaps read as a continuous row rather than four
    # columns stranded on their own with empty gutters.
    bot_gs = fig.add_gridspec(
        1, n_go,
        wspace=0.55,
        left=0.10, right=0.99,
        top=BOT_T, bottom=BOT_B,
    )
    go_row_first_ax = None

    for panel_i, (ds_name, title, tissue) in enumerate(go_datasets):
        csv = go_base / f"go_{ds_name}_all.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)

        rows = []
        for k in sorted(df["latent_dim"].unique()):
            sub = df[df["latent_dim"] == k].sort_values(
                "Adjusted P-value").head(top_k_go)
            for _, r in sub.iterrows():
                rows.append({
                    "latent_dim": k,
                    "term_short": _wrap_term(r["Term"], width=22, max_lines=3),
                    "adj_p": r["Adjusted P-value"],
                })
        if not rows:
            continue
        plot_df = pd.DataFrame(rows)
        plot_df["neg_log_p"] = -np.log10(plot_df["adj_p"].clip(lower=1e-30))

        dims = sorted(plot_df["latent_dim"].unique())
        unique_terms = []
        for k in dims:
            for t in plot_df[plot_df["latent_dim"] == k]["term_short"]:
                if t not in unique_terms:
                    unique_terms.append(t)

        matrix = np.full((len(unique_terms), len(dims)), np.nan)
        for _, row in plot_df.iterrows():
            i = unique_terms.index(row["term_short"])
            j = dims.index(row["latent_dim"])
            matrix[i, j] = row["neg_log_p"]

        ax = fig.add_subplot(bot_gs[0, panel_i])
        if panel_i == 0:
            go_row_first_ax = ax
        vmin = np.nanmin(matrix)
        vmax = np.nanmax(matrix)
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                       vmin=vmin, vmax=vmax)

        mean_v = np.nanmean(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isnan(matrix[i, j]):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               facecolor="#F0F0F0",
                                               edgecolor="none", zorder=2))
                else:
                    v = matrix[i, j]
                    ax.text(j, i, f"{v:.0f}",
                            ha="center", va="center", fontsize=6.5,
                            color="white" if v > mean_v else "#222")

        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels([f"$L_{{{d}}}$" for d in dims], fontsize=9)
        ax.set_yticks(range(len(unique_terms)))
        ax.set_yticklabels(unique_terms, fontsize=6.5, linespacing=0.95)
        ax.set_xlabel("Latent dim", fontsize=9)
        ax.set_title(f"{title}\n{tissue}", fontsize=10, pad=6,
                     fontweight="normal")

        ax.set_xticks(np.arange(-0.5, len(dims), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(unique_terms), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(r"$-\log_{10}$ adj. p", fontsize=7)

    # Single row-level label (f) centred on the GO heatmap row.
    if go_row_first_ax is not None:
        _row_label_from_axes(fig, go_row_first_ax, "f")

    fig.suptitle(
        "Downstream analysis: trajectory / pseudotime and GO-term "
        "enrichment of GAHIB latent dimensions",
        fontsize=13, fontweight="normal", y=0.99)

    if output_path:
        fig.savefig(str(output_path), format="pdf")
    return fig


def generate_downstream_figure(results_base: str | Path | None = None):
    if results_base is None:
        results_base = Path(__file__).resolve().parent.parent.parent / "GAHIB_results"
    results_base = Path(results_base)

    trajectory_base = results_base / "trajectory"
    go_base = results_base / "interpretation" / "go_enrichment"

    trajectory_datasets = [
        ("dentate",        "Dentate gyrus"),
        ("hemato",         "Hematopoiesis"),
        ("setty",          "Setty HSPC"),
        ("hESC_GSE144024", "hESC"),
    ]
    go_datasets = [
        ("dentate",        "Dentate gyrus",      "(mouse, neural)"),
        ("hemato",         "Hematopoiesis",      "(mouse, blood)"),
        ("setty",          "Setty HSPC",         "(human, immune)"),
        ("hESC_GSE144024", "hESC",               "(human, stem)"),
    ]

    out_dir = results_base / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_downstream_analysis.pdf"
    try:
        fig = fig_downstream_analysis(
            trajectory_base, go_base,
            trajectory_datasets, go_datasets,
            output_path=out_path,
        )
        plt.close(fig)
        print(f"  ✓ {out_path}")
    except Exception as e:
        import traceback
        print(f"  ✗ {e}")
        traceback.print_exc()


if __name__ == "__main__":
    generate_downstream_figure()
