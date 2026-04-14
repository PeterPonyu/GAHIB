#!/usr/bin/env python3
"""
GO Biological Process enrichment visualisation for GAHIB latent dimensions.

Reads the combined GO results from run_go_enrichment.py and produces a
publication-quality heatmap: top-5 significant terms × latent dim,
coloured by −log10(adjusted p-value).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd


# Arial font registration (matches fig_new_experiments.py)
_FONT_DIR = Path(__file__).resolve().parent / "fonts"
if _FONT_DIR.exists():
    for ttf in _FONT_DIR.glob("*.ttf"):
        try:
            fm.fontManager.addfont(str(ttf))
        except Exception:
            pass

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.28,
    "pdf.fonttype": 42,
}


def _truncate_term(term: str, max_len: int = 45) -> str:
    """Remove GO:xxxx suffix and wrap long terms onto multiple lines
    at word boundaries so the y-axis labels fit inside the heatmap."""
    import textwrap
    if "(GO:" in term:
        term = term.split("(GO:")[0].strip()
    if term.startswith("GOBP_"):
        term = term[len("GOBP_"):].replace("_", " ").title()
    lines = textwrap.wrap(term, width=max_len // 2 if max_len >= 30 else max_len,
                          break_long_words=False)
    if not lines:
        return term
    return "\n".join(lines[:3])


def fig_go_enrichment(results_dir: str | Path,
                      dataset_name: str | None = None,
                      top_k: int = 3,
                      output_path: str | Path | None = None
                      ) -> plt.Figure:
    """Heatmap of top-K GO BP terms per latent dimension.

    Parameters
    ----------
    results_dir : path to GAHIB_results/interpretation/go_enrichment/
    dataset_name : if given, uses that dataset's combined CSV;
                   otherwise uses go_all_datasets.csv aggregated.
    top_k : number of top terms per dimension to display.
    """
    plt.rcParams.update(_RC)
    results_dir = Path(results_dir)

    if dataset_name:
        csv = results_dir / f"go_{dataset_name}_all.csv"
    else:
        csv = results_dir / "go_all_datasets.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Missing {csv}")

    df = pd.read_csv(csv)

    # For each dim, take top-k terms
    rows = []
    for k in sorted(df["latent_dim"].unique()):
        sub = df[df["latent_dim"] == k].copy()
        if dataset_name is None:
            # Aggregate by term across datasets: min adj-p, count datasets
            sub = sub.groupby("Term").agg(
                adj_p=("Adjusted P-value", "min"),
                n_datasets=("dataset", "nunique"),
            ).reset_index()
            sub = sub.sort_values("adj_p").head(top_k)
            for _, r in sub.iterrows():
                rows.append({"latent_dim": k, "Term": r["Term"],
                             "adj_p": r["adj_p"], "n_ds": r["n_datasets"]})
        else:
            sub = sub.sort_values("Adjusted P-value").head(top_k)
            for _, r in sub.iterrows():
                rows.append({"latent_dim": k, "Term": r["Term"],
                             "adj_p": r["Adjusted P-value"], "n_ds": 1})

    if not rows:
        raise ValueError("No significant GO terms found.")

    plot_df = pd.DataFrame(rows)
    plot_df["neg_log_p"] = -np.log10(plot_df["adj_p"].clip(lower=1e-30))
    plot_df["term_short"] = plot_df["Term"].apply(_truncate_term)

    # Pivot: rows = terms, columns = latent dims, values = neg_log_p
    all_terms = []
    for k in sorted(plot_df["latent_dim"].unique()):
        terms_k = plot_df[plot_df["latent_dim"] == k]["term_short"].tolist()
        for t in terms_k:
            if t not in all_terms:
                all_terms.append(t)

    dims = sorted(plot_df["latent_dim"].unique())
    matrix = np.full((len(all_terms), len(dims)), np.nan)
    for _, row in plot_df.iterrows():
        i = all_terms.index(row["term_short"])
        j = dims.index(row["latent_dim"])
        matrix[i, j] = row["neg_log_p"]

    fig_h = max(6.0, 0.28 * len(all_terms) + 1.5)
    fig = plt.figure(figsize=(8.2, fig_h))
    ax = fig.add_subplot(111)

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))

    # Grey out missing values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor="#F0F0F0",
                                           edgecolor="none", zorder=2))
            else:
                val = matrix[i, j]
                ax.text(j, i, f"{val:.1f}",
                        ha="center", va="center",
                        fontsize=7,
                        color="white" if val > np.nanmean(matrix) else "#222")

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([f"$L_{{{d}}}$" for d in dims], fontsize=10)
    ax.set_yticks(range(len(all_terms)))
    ax.set_yticklabels(all_terms, fontsize=8)
    ax.set_xlabel("Latent dimension", fontsize=10)

    title = f"Top-{top_k} GO BP terms per latent dimension"
    if dataset_name:
        title += f"  ({dataset_name})"
    else:
        title += "  (aggregated across datasets)"
    ax.set_title(title, fontsize=12, pad=10, fontweight="normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r"$-\log_{10}$(adj. p-value)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Thin grid
    ax.set_xticks(np.arange(-0.5, len(dims), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(all_terms), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    if output_path:
        fig.savefig(str(output_path), format="pdf")
    return fig


def fig_go_context_specific(results_dir: str | Path,
                             datasets_info: list,
                             top_k: int = 3,
                             output_path: str | Path | None = None
                             ) -> plt.Figure:
    """Side-by-side panels showing tissue-specific GO enrichment.

    datasets_info : list of (csv_name, display_title, tissue_context) tuples.
    """
    plt.rcParams.update(_RC)
    results_dir = Path(results_dir)

    n = len(datasets_info)
    fig = plt.figure(figsize=(5.2 * n, 8.2))
    gs = fig.add_gridspec(1, n, wspace=1.0,
                          left=0.035, right=0.99, top=0.88, bottom=0.06)

    for panel_i, (ds_name, title, tissue) in enumerate(datasets_info):
        csv = results_dir / f"go_{ds_name}_all.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)

        # Top-k per dim
        rows = []
        for k in sorted(df["latent_dim"].unique()):
            sub = df[df["latent_dim"] == k].sort_values("Adjusted P-value").head(top_k)
            for _, r in sub.iterrows():
                rows.append({
                    "latent_dim": k,
                    "term_short": _truncate_term(r["Term"], max_len=35),
                    "adj_p": r["Adjusted P-value"],
                })
        plot_df = pd.DataFrame(rows)
        plot_df["neg_log_p"] = -np.log10(plot_df["adj_p"].clip(lower=1e-30))

        # Build heatmap matrix with unique terms in insertion order
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

        ax = fig.add_subplot(gs[0, panel_i])
        vmin = np.nanmin(matrix)
        vmax = np.nanmax(matrix)
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                       vmin=vmin, vmax=vmax)

        # Grey missing cells + annotate
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
        ax.set_yticklabels(unique_terms, fontsize=7)
        ax.set_xlabel("Latent dim", fontsize=9)
        ax.set_title(f"{title}\n{tissue}", fontsize=10, pad=6, fontweight="normal")

        # Thin grid lines
        ax.set_xticks(np.arange(-0.5, len(dims), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(unique_terms), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(r"$-\log_{10}$ adj. p", fontsize=7)

    fig.suptitle("Tissue-specific GO BP enrichment of GAHIB latent dimensions",
                 fontsize=13, fontweight="normal", y=0.965)

    if output_path:
        fig.savefig(str(output_path), format="pdf")
    return fig


def generate_all_go_figures(results_base: str | Path | None = None):
    if results_base is None:
        results_base = (Path(__file__).resolve().parent.parent.parent
                        / "GAHIB_results" / "interpretation" / "go_enrichment")
    results_base = Path(results_base)

    fig_dir = results_base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Aggregated figure
    try:
        fig = fig_go_enrichment(results_base, dataset_name=None, top_k=3,
                                output_path=fig_dir / "fig_go_enrichment.pdf")
        plt.close(fig)
        print(f"  ✓ aggregated: {fig_dir / 'fig_go_enrichment.pdf'}")
    except Exception as e:
        print(f"  ✗ aggregated: {e}")

    # Per-dataset figures
    for ds in ["dentate", "hemato", "setty", "hESC_GSE144024"]:
        csv = results_base / f"go_{ds}_all.csv"
        if csv.exists():
            try:
                fig = fig_go_enrichment(results_base, dataset_name=ds, top_k=5,
                                        output_path=fig_dir / f"fig_go_{ds}.pdf")
                plt.close(fig)
                print(f"  ✓ {ds}: {fig_dir / f'fig_go_{ds}.pdf'}")
            except Exception as e:
                print(f"  ✗ {ds}: {e}")

    # Tissue-specific side-by-side panel (the key validation figure)
    datasets_info = [
        ("dentate",        "Dentate gyrus",      "(mouse, neural)"),
        ("hemato",         "Hematopoiesis",      "(mouse, blood)"),
        ("setty",          "Setty HSPC",         "(human, immune)"),
        ("hESC_GSE144024", "hESC",               "(human, stem/hemostasis)"),
    ]
    try:
        fig = fig_go_context_specific(
            results_base, datasets_info, top_k=3,
            output_path=fig_dir / "fig_go_context_specific.pdf")
        plt.close(fig)
        print(f"  ✓ context-specific: {fig_dir / 'fig_go_context_specific.pdf'}")
    except Exception as e:
        print(f"  ✗ context-specific: {e}")


if __name__ == "__main__":
    generate_all_go_figures()
