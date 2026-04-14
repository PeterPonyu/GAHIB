#!/usr/bin/env python3
"""
Publication-quality figures for the four robustness/efficiency experiments:
  1. Hyperparameter sensitivity — line plots with 95% CI ribbons
  2. Latent dimension ablation — grouped bars with connected median line
  3. Seed robustness — compact distribution summary
  4. Computational cost — cost comparison + scaling curve

All figures use Arial font (fallback: DejaVu Sans) and a consistent
Wong-inspired colorblind-safe palette matching the main paper figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd


# ── Font registration: Arial from bundled fonts ─────────────────────────────
_FONT_DIR = Path(__file__).resolve().parent / "fonts"
if _FONT_DIR.exists():
    for ttf in _FONT_DIR.glob("*.ttf"):
        try:
            fm.fontManager.addfont(str(ttf))
        except Exception:
            pass

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.labelsize": 10,
    "axes.labelweight": "normal",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.grid": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.28,
    "pdf.fonttype": 42,  # TrueType embedding (not Type 3)
    "ps.fonttype": 42,
}


def _apply_style():
    plt.rcParams.update(_RC)


# ── Palette (Wong colorblind-safe, matches main paper) ──────────────────────
C_BETA    = "#0072B2"   # blue
C_LAM_IB  = "#E69F00"   # orange
C_LAM_HYP = "#009E73"   # green
C_K_NN    = "#CC79A7"   # pink
C_GAHIB   = "#D55E00"   # vermilion
C_BASE    = "#0072B2"
C_VAEIBH  = "#009E73"
C_HIGH    = "#D62728"   # default marker red
C_FILL    = "#F7F7F7"

SWEEP_META = {
    "beta":    {"color": C_BETA,    "label": r"$\beta$ (KL weight)",
                "default": 0.1},
    "lam_ib":  {"color": C_LAM_IB,  "label": r"$\lambda_{\mathrm{ib}}$ (IB weight)",
                "default": 0.5},
    "lam_hyp": {"color": C_LAM_HYP, "label": r"$\lambda_{\mathrm{hyp}}$ (Lorentz weight)",
                "default": 5.0},
    "k_nn":    {"color": C_K_NN,    "label": r"$k$ (kNN neighbours)",
                "default": 15},
}
METRIC_META = {
    "NMI":                        {"label": "NMI",         "direction": "up"},
    "ARI":                        {"label": "ARI",         "direction": "up"},
    "ASW":                        {"label": "ASW",         "direction": "up"},
    "DRE_umap_overall_quality":   {"label": "DRE (UMAP)",  "direction": "up"},
    "LSE_overall_quality":        {"label": "LSE",         "direction": "up"},
}
METRICS_ORDER = list(METRIC_META.keys())


# ============================================================================
# 1. Hyperparameter sensitivity — 4-column × 5-row grid
# ============================================================================

def fig_hyperparam_sensitivity(results_dir: str | Path,
                                output_path: str | Path | None = None
                                ) -> plt.Figure:
    _apply_style()
    results_dir = Path(results_dir)

    # Load per-sweep summaries
    sweeps = {}
    for name in SWEEP_META:
        csv = results_dir / f"sensitivity_{name}_summary.csv"
        if csv.exists():
            sweeps[name] = pd.read_csv(csv)
    if not sweeps:
        raise FileNotFoundError(f"No sensitivity summaries in {results_dir}")

    metrics = [m for m in METRICS_ORDER if any(
        m in df.columns for df in sweeps.values())]
    n_rows = len(metrics)
    n_cols = len(sweeps)

    fig = plt.figure(figsize=(2.85 * n_cols, 1.85 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.45, wspace=0.45,
                          left=0.07, right=0.98, top=0.93, bottom=0.07)

    for col, (sweep_name, df) in enumerate(sweeps.items()):
        meta = SWEEP_META[sweep_name]
        color = meta["color"]
        values = df["sweep_value"].values

        for row, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[row, col])
            if metric not in df.columns:
                ax.axis("off")
                continue

            y = df[metric].values
            # Main line
            ax.plot(values, y, "-", color=color, lw=1.6, alpha=0.95, zorder=2)
            ax.scatter(values, y, s=38, color=color, edgecolor="white",
                       linewidth=0.8, zorder=3)

            # Highlight default value
            default = meta["default"]
            if default in values:
                idx = int(np.where(values == default)[0][0])
                ax.scatter([default], [y[idx]], s=110, color=C_HIGH,
                           marker="*", edgecolor="white", linewidth=0.8,
                           zorder=5)
                ax.axvline(default, color="#888", ls=":", lw=0.7,
                           alpha=0.6, zorder=1)

            # Y-axis padding
            yr = y.max() - y.min()
            pad = yr * 0.15 if yr > 0 else 0.02
            ax.set_ylim(y.min() - pad, y.max() + pad * 1.2)

            ax.set_ylabel(METRIC_META[metric]["label"], fontsize=9,
                          labelpad=3)

            if row == 0:
                ax.set_title(meta["label"], pad=8, fontsize=10.5)
            if row == n_rows - 1:
                ax.set_xlabel("Value", fontsize=9)
            else:
                ax.set_xticklabels([])

            # X-axis: log scale for beta/lam_ib/lam_hyp; linear for k
            if sweep_name != "k_nn" and values.min() > 0 and values.max() / values.min() > 10:
                ax.set_xscale("log")

            ax.tick_params(axis="both", which="both", labelsize=8)

    fig.suptitle("Hyperparameter Sensitivity Analysis",
                 fontsize=13, fontweight="normal", y=0.99)

    if output_path:
        fig.savefig(str(output_path), format="pdf")
    return fig


# ============================================================================
# 2. Latent dimension ablation
# ============================================================================

def fig_latent_dim_ablation(results_dir: str | Path,
                            output_path: str | Path | None = None
                            ) -> plt.Figure:
    _apply_style()
    results_dir = Path(results_dir)
    tables = results_dir / "tables"

    csvs = sorted(tables.glob("latdim_*_df.csv"))
    if not csvs:
        raise FileNotFoundError(f"No latdim CSVs in {tables}")

    all_df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    dims = sorted(all_df["latent_dim"].dropna().astype(int).unique().tolist())

    metrics = [m for m in METRICS_ORDER if m in all_df.columns]

    fig = plt.figure(figsize=(2.9 * len(metrics), 3.4))
    gs = fig.add_gridspec(1, len(metrics), wspace=0.42,
                          left=0.06, right=0.98, top=0.86, bottom=0.16)

    x = np.arange(len(dims))
    width = 0.58

    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        means = [all_df[all_df["latent_dim"] == d][metric].mean() for d in dims]
        stds = [all_df[all_df["latent_dim"] == d][metric].std() for d in dims]

        # Per-dim distribution as violin-lite (scatter jitter)
        for j, d in enumerate(dims):
            vals = all_df[all_df["latent_dim"] == d][metric].dropna().values
            jitter = np.random.RandomState(42).normal(0, 0.05, size=len(vals))
            ax.scatter(np.full_like(vals, j) + jitter, vals,
                       s=5, color=C_GAHIB, alpha=0.25, edgecolor="none",
                       zorder=1)

        bars = ax.bar(x, means, width, yerr=stds, capsize=3,
                      color=C_GAHIB, alpha=0.75, edgecolor="#555",
                      linewidth=0.6,
                      error_kw={"linewidth": 0.8, "ecolor": "#444"},
                      zorder=2)

        # Highlight d=10
        if 10 in dims:
            idx10 = dims.index(10)
            bars[idx10].set_edgecolor(C_HIGH)
            bars[idx10].set_linewidth(1.8)
            bars[idx10].set_alpha(0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel(r"Latent dimension $d$")
        ax.set_title(METRIC_META[metric]["label"])
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="y", linewidth=0.4, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle("Latent Dimension Ablation (53 datasets)",
                 fontsize=13, fontweight="normal", y=0.99)

    if output_path:
        fig.savefig(str(output_path), format="pdf")
    return fig


# ============================================================================
# 3. Seed robustness — compact summary
# ============================================================================

def fig_seed_robustness(results_dir: str | Path,
                        output_path: str | Path | None = None
                        ) -> plt.Figure:
    _apply_style()
    results_dir = Path(results_dir)
    csv = results_dir / "seed_robustness_summary.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Missing {csv}")

    df = pd.read_csv(csv)
    metrics = [m for m in METRICS_ORDER if f"{m}_mean" in df.columns]

    fig = plt.figure(figsize=(2.8 * len(metrics), 3.6))
    gs = fig.add_gridspec(1, len(metrics), wspace=0.35,
                          left=0.05, right=0.98, top=0.84, bottom=0.13)

    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        means = df[f"{metric}_mean"].values
        stds = df[f"{metric}_std"].values
        order = np.argsort(means)
        means = means[order]
        stds = stds[order]
        y = np.arange(len(means))

        # Error bars
        ax.errorbar(means, y, xerr=stds, fmt="none",
                    ecolor="#888", elinewidth=0.6, capsize=1.6, zorder=1)
        # Markers coloured by mean
        ax.scatter(means, y, s=14, c=means, cmap="viridis",
                   edgecolor="white", linewidth=0.4, zorder=3)

        # Overall mean ± std
        overall_mean = float(np.mean(means))
        overall_std = float(np.mean(stds))
        ax.axvline(overall_mean, color=C_HIGH, ls="--", lw=1.0,
                   alpha=0.8, zorder=2)
        ax.text(overall_mean, len(means) * 1.05,
                f"{overall_mean:.3f} ± {overall_std:.4f}",
                fontsize=8, color=C_HIGH, ha="center",
                fontweight="normal")

        ax.set_xlabel(f"{METRIC_META[metric]['label']}  (mean ± std over 5 seeds)",
                      fontsize=9)
        ax.set_title(METRIC_META[metric]["label"])
        ax.set_yticks([])
        ax.set_ylim(-1, len(means) * 1.12)
        ax.grid(axis="x", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Multi-Seed Reproducibility  (5 seeds × 53 datasets)",
                 fontsize=13, fontweight="normal", y=0.99)

    if output_path:
        fig.savefig(str(output_path), format="pdf")
    return fig


# ============================================================================
# 4. Computational cost
# ============================================================================

def fig_computational_cost(results_dir: str | Path,
                           output_path: str | Path | None = None
                           ) -> plt.Figure:
    _apply_style()
    results_dir = Path(results_dir)

    cost_csv = results_dir / "cost_summary.csv"
    scale_csv = results_dir / "scaling_summary.csv"

    fig = plt.figure(figsize=(10.0, 3.8))
    gs = fig.add_gridspec(1, 2, wspace=0.30,
                          left=0.07, right=0.98, top=0.84, bottom=0.17)

    # ── Panel A: method cost comparison ──
    if cost_csv.exists():
        ax = fig.add_subplot(gs[0, 0])
        cost_df = pd.read_csv(cost_csv, header=[0, 1], index_col=0)
        methods = cost_df.index.tolist()
        t_mean = cost_df[("train_time", "mean")].values
        t_std = cost_df[("train_time", "std")].values
        m_mean = cost_df[("peak_memory_gb", "mean")].values

        color_map = {"GAHIB": C_GAHIB, "Base VAE": C_BASE, "VAE+IB+Hyp": C_VAEIBH}
        colors = [color_map.get(m, "#888") for m in methods]
        x = np.arange(len(methods))
        bars = ax.bar(x, t_mean, 0.55, yerr=t_std, capsize=4,
                      color=colors, alpha=0.88, edgecolor="#333",
                      linewidth=0.6,
                      error_kw={"linewidth": 0.9, "ecolor": "#333"})

        # Memory annotation
        ymax = (t_mean + t_std).max()
        for bar, mem in zip(bars, m_mean):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (ymax * 0.02),
                    f"{mem * 1000:.0f} MB",
                    ha="center", va="bottom",
                    fontsize=8.5, color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("Training time (s)")
        ax.set_title("(a) Per-dataset cost  (53 datasets, 200 epochs)")
        ax.set_ylim(0, ymax * 1.18)
        ax.grid(axis="y", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)

    # ── Panel B: scaling curve ──
    if scale_csv.exists():
        ax = fig.add_subplot(gs[0, 1])
        sdf = pd.read_csv(scale_csv)
        grouped = sdf.groupby("target_cells").agg(
            mean=("train_time", "mean"),
            std=("train_time", "std"),
            n=("train_time", "count"),
        ).reset_index()
        cells = grouped["target_cells"].values
        y_mean = grouped["mean"].values
        y_std = grouped["std"].values

        ax.fill_between(cells, y_mean - y_std, y_mean + y_std,
                        color=C_GAHIB, alpha=0.18, zorder=1)
        ax.plot(cells, y_mean, "o-", color=C_GAHIB, lw=2.0,
                markersize=8, mec="white", mew=1.2, zorder=3)

        # Individual points as scatter
        for _, row in sdf.iterrows():
            ax.scatter([row["target_cells"]], [row["train_time"]],
                       s=14, color=C_GAHIB, alpha=0.35, edgecolor="none",
                       zorder=2)

        ax.set_xlabel("Number of cells")
        ax.set_ylabel("Training time (s)")
        ax.set_title("(b) GAHIB scaling  (10 datasets × 4 sizes)")
        ax.set_xticks(list(cells))
        ax.set_xlim(min(cells) * 0.9, max(cells) * 1.05)
        ax.grid(True, linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Computational Cost Analysis",
                 fontsize=13, fontweight="normal", y=0.99)

    if output_path:
        fig.savefig(str(output_path), format="pdf")
    return fig


# ============================================================================
# Entry point
# ============================================================================

def generate_all_new_figures(results_base: str | Path | None = None):
    if results_base is None:
        results_base = Path(__file__).resolve().parent.parent.parent / "GAHIB_results"
    results_base = Path(results_base)

    experiments = [
        ("hyperparam_sensitivity", fig_hyperparam_sensitivity,
         "fig_hyperparam_sensitivity.pdf"),
        ("latent_dim_ablation", fig_latent_dim_ablation,
         "fig_latent_dim_ablation.pdf"),
        ("seed_robustness", fig_seed_robustness,
         "fig_seed_robustness.pdf"),
        ("computational_cost", fig_computational_cost,
         "fig_computational_cost.pdf"),
    ]
    for exp_name, fig_func, fname in experiments:
        exp_dir = results_base / exp_name
        fig_dir = exp_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        try:
            fig = fig_func(exp_dir, fig_dir / fname)
            plt.close(fig)
            print(f"  ✓ {exp_name}: {fig_dir / fname}")
        except Exception as e:
            print(f"  ✗ {exp_name}: {e}")


if __name__ == "__main__":
    generate_all_new_figures()
