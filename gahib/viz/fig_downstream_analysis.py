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


def _panel_label(ax, label, x=-0.18, y=1.06, fontsize=14):
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            va="bottom", ha="left")


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


def _load_trajectory_panel(tables_dir: Path, ds_name: str):
    """Return (umap_xy, labels, gahib_pt, top_neg, top_pos, abs_mean_top20)
    or None if not reconstructible."""
    from .fig_trajectory import _maybe_recompute_gene_panel
    npz_file = tables_dir / f"traj_{ds_name}_pts.npz"
    if not npz_file.exists():
        return None
    data = np.load(str(npz_file), allow_pickle=True)
    recovered = _maybe_recompute_gene_panel(ds_name, data)
    if recovered is None:
        return None
    umap_xy, top_neg, top_pos, abs_mean = recovered
    return (umap_xy, data["labels"], data["gahib_pt"],
            top_neg, top_pos, abs_mean)


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

    # Top region: trajectory panels (rows a..d and summary e)
    top_gs = fig.add_gridspec(
        5, n_traj,
        height_ratios=[1.0, 1.0, 1.0, 1.0, 0.85],
        hspace=0.62, wspace=0.40,
        left=0.06, right=0.97,
        top=0.965, bottom=0.545,
    )

    gene_summary_data = []

    for col, (ds_name, title) in enumerate(trajectory_datasets):
        rec = _load_trajectory_panel(tables_dir, ds_name)
        if rec is None:
            continue
        umap_xy, labels, gahib_pt, top_neg, top_pos, abs_mean = rec
        gene_summary_data.append((title, abs_mean))

        # Row 1: clusters
        ax = fig.add_subplot(top_gs[0, col])
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
        ax.set_title(f"{title}\nLeiden clusters" if col == 0 else title,
                     fontsize=10, pad=4)
        if col == 0:
            _panel_label(ax, "a")

        # Row 2: GAHIB pt
        ax = fig.add_subplot(top_gs[1, col])
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

        # Row 3: top negative gene
        gname, gvec, gr = top_neg
        ax = fig.add_subplot(top_gs[2, col])
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
            ax.set_title("Early / progenitor-like gene\n(expr. ↓ with pt)",
                         fontsize=10, pad=4)
            _panel_label(ax, "c")

        # Row 4: top positive gene
        gname, gvec, gr = top_pos
        ax = fig.add_subplot(top_gs[3, col])
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
            ax.set_title("Lineage-commitment gene\n(expr. ↑ with pt)",
                         fontsize=10, pad=4)
            _panel_label(ax, "d")

    # Row 5 (summary bar spans all trajectory columns)
    ax = fig.add_subplot(top_gs[4, :])
    if gene_summary_data:
        titles, vals = zip(*gene_summary_data)
        x = np.arange(len(titles))
        ax.bar(x, vals, width=0.55, color="#D55E00",
               alpha=0.88, edgecolor="#333", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(titles, fontsize=9)
        ax.set_ylabel("Mean |Pearson r| of top-20 HVGs", fontsize=9)
        ax.set_title(
            "Biological alignment of GAHIB pseudotime with top gene programmes",
            fontsize=10.5, pad=6)
        ax.set_ylim(0, max(1.0, max(vals) * 1.15))
        ax.grid(axis="y", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)
        _panel_label(ax, "e", x=-0.045, y=1.10)

    # ── Bottom region: tissue-specific GO enrichment panels ──
    # Wrapped GO term labels free up horizontal space; tighten wspace
    # so the four heatmaps read as a continuous row rather than four
    # columns stranded on their own with empty gutters.
    bot_gs = fig.add_gridspec(
        1, n_go,
        wspace=0.55,
        left=0.06, right=0.99,
        top=0.475, bottom=0.07,
    )

    panel_letters = list("fghijklmn")

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

        if panel_i < len(panel_letters):
            _panel_label(ax, panel_letters[panel_i], x=-0.28, y=1.05)

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
