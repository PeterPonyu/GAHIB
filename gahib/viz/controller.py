"""
GAHIB Automatic Visualization Controller
==========================================

Fully automatic system for loading, computing, and visualizing metrics from
the GAHIB benchmark results. Integrates the DRE (Dimensionality Reduction
Evaluator) and LSE (Latent Space Evaluator) series from the MoCoO evaluation
framework.

Usage
-----
    from gahib.viz.controller import VisualizationController

    ctrl = VisualizationController(results_dir="GAHIB_results/ablation/tables")
    ctrl.load_all()
    ctrl.generate_all_figures(output_dir="figures/ablation")
"""
from __future__ import annotations

import os
import glob
import json
import logging
import string
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from scipy import stats as scipy_stats

from . import style as S

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric group definitions for figure generation
# ---------------------------------------------------------------------------
METRIC_GROUPS = {
    "clustering": {
        "metrics": S.METRICS_CLUSTERING,
        "title": "Clustering Metrics",
        "ncols": 6,
        "figsize": (19.2, 3.8),
    },
    "dre_umap": {
        "metrics": S.METRICS_DRE_UMAP,
        "title": "DRE Series (UMAP)",
        "ncols": 4,
        "figsize": (12.8, 3.8),
    },
    "dre_tsne": {
        "metrics": S.METRICS_DRE_TSNE,
        "title": "DRE Series (t-SNE)",
        "ncols": 4,
        "figsize": (12.8, 3.8),
    },
    "lse": {
        "metrics": S.METRICS_LSE,
        "title": "LSE Series (Intrinsic Latent Quality)",
        "ncols": 6,
        "figsize": (19.2, 3.8),
    },
}


class VisualizationController:
    """Fully automatic visualization system for GAHIB benchmark results.

    Loads CSV result tables from a directory, extracts DRE and LSE series
    metrics, and generates publication-quality figures matching the MoCoO
    article style (IEEE J-BHI, 17x21cm, 300 DPI, Arial).

    Parameters
    ----------
    results_dir : str or Path
        Directory containing per-dataset CSV files (e.g., ablation_dentate_df.csv)
    method_names : list of str, optional
        Names of methods in the CSV index. Auto-detected if None.
    method_order : list of str, optional
        Display order. Defaults to method_names order.
    palette : dict or list, optional
        Color mapping. Defaults to GAHIB palette.
    """

    def __init__(
        self,
        results_dir: str | Path,
        method_names: Optional[List[str]] = None,
        method_order: Optional[List[str]] = None,
        palette: Optional[dict | list] = None,
    ):
        self.results_dir = Path(results_dir)
        self.method_names = self._normalize_method_list(method_names)
        self.method_order = self._normalize_method_list(method_order)
        self.palette = palette

        # Data storage
        self.raw_data: Dict[str, pd.DataFrame] = {}  # dataset_name -> DataFrame
        self.long_data: Optional[pd.DataFrame] = None  # melted long-form
        self.summary: Optional[pd.DataFrame] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> "VisualizationController":
        """Scan results_dir for CSV files and load all metrics data."""
        S.apply_style()

        csv_files = sorted(glob.glob(str(self.results_dir / "*.csv")))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.results_dir}"
            )

        for fpath in csv_files:
            fname = os.path.basename(fpath)
            # Extract dataset name from filename patterns like:
            #   ablation_dentate_df.csv -> dentate
            #   benchmark_BoneMarrow_df.csv -> BoneMarrow
            name = fname
            for prefix in ["ablation_", "benchmark_", "gmvae_", "disent_"]:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            name = name.replace("_df.csv", "").replace(".csv", "")

            try:
                df = pd.read_csv(fpath, index_col=0)
                # Apply legacy name mapping for backward compatibility
                df.index = [S.LEGACY_NAME_MAP.get(str(n), str(n))
                            for n in df.index]
                # Clean NaN with column medians
                for col in df.columns:
                    if df[col].isna().any():
                        median_val = df[col].median()
                        df[col] = df[col].fillna(
                            median_val if not pd.isna(median_val) else 0
                        )
                self.raw_data[name] = df
            except Exception as e:
                warnings.warn(f"Failed to load {fpath}: {e}")

        if not self.raw_data:
            raise ValueError("No valid CSV data loaded")

        # Auto-detect methods
        first_df = next(iter(self.raw_data.values()))
        if self.method_names is None:
            self.method_names = list(first_df.index)
        if self.method_order is None:
            self.method_order = list(self.method_names)
        self.method_names = self._normalize_method_list(self.method_names)
        self.method_order = self._normalize_method_list(self.method_order)
        if self.palette is None:
            config_colors = S.get_config_colors()
            self.palette = {
                m: config_colors.get(m, S._PALETTE[i % len(S._PALETTE)])
                for i, m in enumerate(self.method_names)
            }

        # Build long-form DataFrame for plotting
        self._build_long_data()
        self._build_summary()
        self._loaded = True

        logger.info("Loaded %d datasets, %d methods",
                     len(self.raw_data), len(self.method_names))
        return self

    def _build_long_data(self):
        """Melt raw DataFrames into a single long-form DataFrame."""
        records = []
        for dataset, df in self.raw_data.items():
            for method in df.index:
                for metric in df.columns:
                    val = df.loc[method, metric]
                    if np.isscalar(val) and not pd.isna(val):
                        records.append({
                            "dataset": dataset,
                            "method": str(method),
                            "metric": metric,
                            "value": float(val),
                        })
        self.long_data = pd.DataFrame(records)

    def _build_summary(self):
        """Compute per-method mean +/- std across datasets."""
        if self.long_data is None or self.long_data.empty:
            return

        summary = (
            self.long_data
            .groupby(["method", "metric"])["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        self.summary = summary

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    def compute_significance(
        self,
        metric: str,
        method_a: str,
        method_b: str,
    ) -> Tuple[float, str]:
        """Compute Wilcoxon signed-rank test between two methods on a metric."""
        if self.long_data is None:
            return 1.0, "ns"

        vals_a = []
        vals_b = []
        for dataset in self.raw_data:
            df = self.raw_data[dataset]
            if method_a in df.index and method_b in df.index and metric in df.columns:
                va = df.loc[method_a, metric]
                vb = df.loc[method_b, metric]
                if np.isfinite(va) and np.isfinite(vb):
                    vals_a.append(va)
                    vals_b.append(vb)

        if len(vals_a) < 3:
            return 1.0, "ns"

        try:
            stat, pval = scipy_stats.wilcoxon(vals_a, vals_b)
        except Exception:
            return 1.0, "ns"

        if pval < 0.001:
            return pval, "***"
        elif pval < 0.01:
            return pval, "**"
        elif pval < 0.05:
            return pval, "*"
        return pval, "ns"

    def sort_methods_by_performance_gap(self, focal_method: str = "GAHIB"):
        """Reorder methods by descending performance gap from the focal method.

        Methods with the largest gap appear on the left; the focal method
        is placed on the far right.  Uses mean normalised rank across all
        available metrics (higher-is-better after direction correction).
        """
        if self.summary is None or self.summary.empty:
            return

        available = self.long_data["metric"].unique()
        method_scores: Dict[str, float] = {}

        for method in self.method_order:
            scores = []
            for metric in available:
                mask = (
                    (self.summary["method"] == method)
                    & (self.summary["metric"] == metric)
                )
                vals = self.summary.loc[mask, "mean"]
                if vals.empty:
                    continue
                v = vals.values[0]
                # Invert metrics where lower is better so that a higher
                # aggregated score always means "better".
                if not S.METRIC_DIRECTION.get(metric, True):
                    v = -v
                scores.append(v)
            method_scores[method] = np.mean(scores) if scores else -np.inf

        # Sort ascending (worst first), but force focal method to end
        non_focal = [m for m in self.method_order if m != focal_method]
        non_focal.sort(key=lambda m: method_scores.get(m, -np.inf))
        if focal_method in self.method_order:
            self.method_order = non_focal + [focal_method]
        else:
            self.method_order = non_focal

    # ------------------------------------------------------------------
    # Figure generation
    # ------------------------------------------------------------------

    def generate_all_figures(
        self,
        output_dir: str | Path,
        sig_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, str]:
        """Generate all metric group figures and save to output_dir.

        Returns a dict mapping group name to output file path.
        """
        self._ensure_loaded()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if sig_pairs is None and len(self.method_names) >= 2:
            # Default: compare everything against the last method (full model)
            full = self.method_names[-1]
            sig_pairs = [(m, full) for m in self.method_names[:-1]]

        results = {}
        available_groups = self._get_available_groups()

        for group_name, group_cfg, available in available_groups:

            try:
                outpath = output_dir / f"fig_{group_name}.pdf"
                fig = self._plot_metric_group(
                    metrics=available,
                    title=group_cfg["title"],
                    ncols=min(len(available), group_cfg["ncols"]),
                    figsize=group_cfg.get("figsize"),
                    sig_pairs=sig_pairs,
                )
                S.save_figure(fig, str(outpath))
                plt.close(fig)
                results[group_name] = str(outpath)
                logger.info("Generated: %s", outpath)
            except Exception as e:
                warnings.warn(f"Failed to generate {group_name}: {e}")
                logger.debug("Traceback for %s failure:", group_name, exc_info=True)

        try:
            outpath = output_dir / "fig_all_metrics.pdf"
            fig = self._plot_composed_metric_groups(
                available_groups=available_groups,
                sig_pairs=sig_pairs,
            )
            S.save_figure(fig, str(outpath))
            plt.close(fig)
            results["all_metrics"] = str(outpath)
            logger.info("Generated: %s", outpath)
        except Exception as e:
            warnings.warn(f"Failed to generate composed metrics figure: {e}")
            logger.debug("Traceback for composed metrics failure:", exc_info=True)

        # Summary heatmap
        try:
            outpath = output_dir / "fig_summary_heatmap.pdf"
            fig = self._plot_summary_heatmap()
            S.save_figure(fig, str(outpath))
            plt.close(fig)
            results["summary_heatmap"] = str(outpath)
            logger.info("Generated: %s", outpath)
        except Exception as e:
            warnings.warn(f"Failed to generate summary heatmap: {e}")

        # Mean +/- std table
        try:
            table_path = output_dir / "mean_std_table.csv"
            self._save_summary_table(table_path)
            results["summary_table"] = str(table_path)
            logger.info("Generated: %s", table_path)
        except Exception as e:
            warnings.warn(f"Failed to save summary table: {e}")

        # LaTeX table
        try:
            tex_path = output_dir / "mean_std_table.tex"
            self._save_latex_table(tex_path)
            results["latex_table"] = str(tex_path)
            logger.info("Generated: %s", tex_path)
        except Exception as e:
            warnings.warn(f"Failed to save LaTeX table: {e}")

        # Statistical significance summary
        if sig_pairs:
            try:
                sig_path = output_dir / "statistical_summary.csv"
                self._save_significance_table(sig_path, sig_pairs)
                results["significance"] = str(sig_path)
                logger.info("Generated: %s", sig_path)
            except Exception as e:
                warnings.warn(f"Failed to save significance: {e}")

        return results

    def _plot_metric_group(
        self,
        metrics: List[str],
        title: str,
        ncols: int,
        figsize: Optional[Tuple[float, float]] = None,
        sig_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        """Plot a group of metrics as side-by-side boxplots."""
        n_metrics = len(metrics)
        if figsize is None:
            figsize = (3.2 * n_metrics, 3.8)

        fig = plt.figure(figsize=figsize)
        axes = S.row_of_axes(fig, n_metrics, S.RECT_BOXPLOT_ROW, gap=S.GAP_BOXPLOT)
        self._populate_metric_axes(
            axes=axes,
            metrics=metrics,
            sig_pairs=sig_pairs,
        )

        fig.text(0.5, S.RECT_TITLE_Y, title,
                 fontsize=S.FS_TITLE + 2, fontweight="normal",
                 ha="center", va="bottom")
        return fig

    def _populate_metric_axes(
        self,
        axes,
        metrics,
        sig_pairs=None,
        panel_label=None,
        font_scale: float = 1.0,
        ytick_scale: float = 1.0,
        first_col_only_yticks: bool = False,
    ):
        methods_present_cache = None

        for i, metric in enumerate(metrics):
            ax = axes[i]
            plot_data = self.long_data[
                self.long_data["metric"] == metric
            ].copy()

            if plot_data.empty:
                ax.set_title(S.metric_title(metric), fontsize=S.FS_TITLE * font_scale)
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center")
                continue

            # Filter to known methods and enforce order
            plot_data = plot_data[
                plot_data["method"].isin(self.method_order)
            ]
            plot_data["method"] = pd.Categorical(
                plot_data["method"],
                categories=self.method_order,
                ordered=True,
            )

            # Boxplot + stripplot
            palette_list = [self.palette.get(m, "#999999")
                            for m in self.method_order
                            if m in plot_data["method"].values]
            methods_present = [m for m in self.method_order
                               if m in plot_data["method"].values]
            methods_present_cache = methods_present

            sns.boxplot(
                data=plot_data, x="method", y="value",
                hue="method",
                order=methods_present,
                hue_order=methods_present,
                palette=palette_list,
                width=0.6, linewidth=0.8, fliersize=2,
                dodge=False,
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=plot_data, x="method", y="value",
                order=methods_present,
                color="black", size=4, alpha=0.5,
                jitter=0.35, ax=ax,
            )

            ax.set_title(S.metric_title(metric), fontsize=S.FS_TITLE * font_scale)
            ax.set_xlabel("")
            ax.set_ylabel("")
            self._format_x_axis(ax, methods_present, font_scale=font_scale)
            # Always keep y tick labels visible per metric panel so each
            # subplot communicates its own y-range.
            ax.tick_params(axis="y", labelsize=S.FS_TICK * ytick_scale)

            # Cleaner dense layout: remove right/top spines so long y ticks do
            # not visually collide with neighbor panel right borders.
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            self._format_y_axis(ax, plot_data["value"].to_numpy())

            # Significance brackets
            if sig_pairs:
                self._add_significance_brackets(ax, metric, methods_present,
                                                sig_pairs, plot_data)

        if panel_label and axes:
            S.add_panel_label(axes[0], panel_label, x=-0.12, y=1.06,
                              fontsize=S.FS_LABEL)

    def _add_significance_brackets(self, ax, metric, methods, sig_pairs, data):
        """Add significance brackets between pairs."""
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        bracket_y = ymax + y_range * 0.04
        step = y_range * 0.10 if y_range > 0 else 0.05

        valid_pairs = []
        for ma, mb in sig_pairs:
            if ma not in methods or mb not in methods:
                continue
            pval, stars = self.compute_significance(metric, ma, mb)
            if stars == "ns":
                continue

            x1 = methods.index(ma)
            x2 = methods.index(mb)
            if x1 == x2:
                continue
            if x1 > x2:
                x1, x2 = x2, x1
            valid_pairs.append((x1, x2, stars))

        valid_pairs.sort(key=lambda item: (item[1] - item[0], item[0]))

        bracket_count = 0
        for x1, x2, stars in valid_pairs:
            y = bracket_y + bracket_count * step

            arm_h = step * 0.20
            ax.plot([x1, x1, x2, x2],
                    [y, y + arm_h, y + arm_h, y],
                    lw=1.2, c="black")
            # Anchor the stars to the bracket arm (centered on the arm
            # horizontally, baseline flush with the arm itself).
            ax.text((x1 + x2) / 2, y + arm_h * 0.1, stars,
                    ha="center", va="bottom",
                    fontsize=S.FS_TITLE + 2,
                    fontweight="normal")
            bracket_count += 1

        if bracket_count > 0:
            ax.set_ylim(ymin, bracket_y + (bracket_count + 0.5) * step)

    def _plot_composed_metric_groups(self, available_groups, sig_pairs=None):
        """Stack all metric groups into a single composed figure."""
        if not available_groups:
            raise ValueError("No metric groups available for composition")

        available_metrics = set(self.long_data["metric"].values)

        # Keep a consistent 4x5 layout (20 metrics total):
        # - remove LSE_core_quality
        # - reduce side whitespace by ensuring each row has 5 panels.
        row_specs = [
            {
                "title": "Clustering Metrics",
                "metrics": ["NMI", "ARI", "ASW", "DAV", "CAL"],
            },
            {
                "title": "COR + DRE Series (UMAP)",
                "metrics": [
                    "COR",
                    "DRE_umap_distance_correlation",
                    "DRE_umap_Q_local",
                    "DRE_umap_Q_global",
                    "DRE_umap_overall_quality",
                ],
            },
            {
                "title": "DRE Series (t-SNE) + Man. dim.",
                "metrics": [
                    "DRE_tsne_distance_correlation",
                    "DRE_tsne_Q_local",
                    "DRE_tsne_Q_global",
                    "DRE_tsne_overall_quality",
                    "LSE_manifold_dimensionality",
                ],
            },
            {
                "title": "LSE Intrinsic Metrics",
                "metrics": [
                    "LSE_spectral_decay_rate",
                    "LSE_participation_ratio",
                    "LSE_anisotropy_score",
                    "LSE_noise_resilience",
                    "LSE_overall_quality",
                ],
            },
        ]
        row_specs = [
            {
                "title": row["title"],
                "metrics": [m for m in row["metrics"] if m in available_metrics],
            }
            for row in row_specs
        ]
        row_specs = [row for row in row_specs if row["metrics"]]

        n_rows = len(row_specs)
        max_cols = max(len(row["metrics"]) for row in row_specs)
        fig_w = max(19.2, max_cols * 3.05)

        # ── Inch-based slot layout ─────────────────────────────────────────
        # Each row slot = TITLE_IN (group title above axes) + AXES_IN
        #               + XTICK_IN (rotated x-tick reserve below axes).
        # Matplotlib subplot titles extend ~0.15 in above axes_top, so
        # TITLE_IN must be large enough that the group title center clears
        # that zone: center at TITLE_IN * 0.40 from top ≥ 0.30 in above
        # axes_top.  TITLE_IN = 0.60 in → center at 0.36 in → clear.
        HEADER_IN = 0.44    # overall figure title zone
        TITLE_IN  = 0.68    # per-row group title zone (above axes)
        AXES_IN   = 2.80    # axes plot area height
        XTICK_IN  = 0.75    # below-axes space for rotated x-tick labels
        GAP_IN    = 0.40    # gap between consecutive row slots
        FOOTER_IN = 0.20    # bottom breathing room

        slot_in = TITLE_IN + AXES_IN + XTICK_IN
        fig_h   = (HEADER_IN + FOOTER_IN
                   + n_rows * slot_in
                   + (n_rows - 1) * GAP_IN)
        fig_h   = max(14.0, fig_h)

        def frac(inches):
            return inches / fig_h

        fig = plt.figure(figsize=(fig_w, fig_h))

        full_left = 0.030
        full_w    = 0.952
        panel_labels = iter(string.ascii_lowercase)

        # Advance cursor top-down; starts just below the header zone
        cursor = 1.0 - frac(HEADER_IN)

        for row_idx, row in enumerate(row_specs):
            if row_idx > 0:
                cursor -= frac(GAP_IN)

            slot_top    = cursor
            # Group title: centred at 40 % from the top of the title zone
            # (= 60 % from axes_top), keeping it clear of subplot titles
            # that matplotlib places ~0.15 in above the axes boundary.
            title_cy    = slot_top - frac(TITLE_IN * 0.40)
            axes_top    = slot_top - frac(TITLE_IN)
            axes_bottom = axes_top  - frac(AXES_IN)
            # Advance cursor past the x-tick reserve
            cursor      = axes_bottom - frac(XTICK_IN)

            n_metrics = len(row["metrics"])
            row_ratio = n_metrics / max_cols
            row_w     = full_w * (0.97 * row_ratio + 0.03)
            row_left  = full_left + (full_w - row_w) / 2

            axes = S.row_of_axes(
                fig,
                n_metrics,
                [row_left, axes_bottom, row_w, frac(AXES_IN)],
                gap=0.018,
            )
            self._populate_metric_axes(
                axes=axes,
                metrics=row["metrics"],
                sig_pairs=sig_pairs,
                panel_label=next(panel_labels),
                font_scale=1.26,
                ytick_scale=1.00,
                first_col_only_yticks=False,
            )
            fig.text(
                0.5,
                title_cy,
                row["title"],
                fontsize=S.FS_TITLE + 3,
                fontweight="normal",
                ha="center",
                va="center",
            )

        # Overall figure title centred in the HEADER_IN zone
        fig.text(
            0.5, 1.0 - frac(HEADER_IN * 0.40),
            "GAHIB Benchmark Metrics Overview",
            fontsize=S.FS_TITLE + 5, fontweight="normal",
            ha="center", va="center",
        )
        return fig

    def _plot_summary_heatmap(self):
        """Generate a summary heatmap of mean metric values across methods."""
        if self.summary is None:
            raise ValueError("No summary data available")

        # Use proposed metrics
        proposed = [m for m in S.PROPOSED_METRICS
                    if m in self.long_data["metric"].values]
        if not proposed:
            proposed = [m for m in S.METRICS_CLUSTERING
                        if m in self.long_data["metric"].values]

        # Build matrix
        methods = self.method_order
        matrix = np.full((len(methods), len(proposed)), np.nan)

        for j, metric in enumerate(proposed):
            for i, method in enumerate(methods):
                mask = (
                    (self.summary["method"] == method) &
                    (self.summary["metric"] == metric)
                )
                vals = self.summary.loc[mask, "mean"]
                if not vals.empty:
                    matrix[i, j] = vals.values[0]

        figsize = (len(proposed) * 1.5 + 2, len(methods) * 0.8 + 1.5)
        fig = plt.figure(figsize=figsize)
        ax = S.place_axes(fig, S.RECT_HEATMAP)

        # Normalize per column for color mapping
        norm_matrix = np.copy(matrix)
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                vmin, vmax = valid.min(), valid.max()
                if vmax > vmin:
                    norm_matrix[:, j] = (col - vmin) / (vmax - vmin)
                    # Invert for DAV (lower is better)
                    metric_name = proposed[j]
                    if metric_name in S.METRIC_DIRECTION and not S.METRIC_DIRECTION[metric_name]:
                        norm_matrix[:, j] = 1.0 - norm_matrix[:, j]

        im = ax.imshow(norm_matrix, cmap=S.HEATMAP_CMAP, aspect="auto",
                       vmin=0, vmax=1)

        # Annotate cells with actual values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                nval = norm_matrix[i, j]
                color = "white" if nval > S.HEATMAP_DARK_THRESHOLD else "black"

                fmt = S.FMT_LARGE if abs(val) > 10 else S.FMT_SCORE_SHORT
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=S.FS_SMALL, color=color, fontweight="normal")

        # Labels
        display_labels = [S.METRIC_DISPLAY.get(m, m) for m in proposed]
        ax.set_xticks(range(len(proposed)))
        ax.set_xticklabels(display_labels, rotation=45, ha="right",
                           fontsize=S.FS_TICK)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(
            [S.get_display_name(m) for m in methods],
            fontsize=S.FS_TICK,
        )
        ax.set_title("GAHIB Benchmark Summary", fontsize=S.FS_TITLE + 1,
                      fontweight="normal", pad=10)

        cbar_rect = [S.RECT_HEATMAP[0] + S.RECT_HEATMAP[2] + 0.02,
                     S.RECT_HEATMAP[1], 0.025, S.RECT_HEATMAP[3]]
        cax = fig.add_axes(cbar_rect)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Normalized score", fontsize=S.FS_AXIS)

        return fig

    def _save_summary_table(self, path: Path):
        """Save mean +/- std table as CSV."""
        if self.summary is None:
            return

        pivot_mean = self.summary.pivot(
            index="method", columns="metric", values="mean"
        )
        pivot_std = self.summary.pivot(
            index="method", columns="metric", values="std"
        )

        # Combine into "mean +/- std" format
        combined = pivot_mean.copy()
        for col in combined.columns:
            if col in pivot_std.columns:
                combined[col] = combined[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else ""
                ) + " +/- " + pivot_std[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else ""
                )

        combined.to_csv(path)

    def _save_latex_table(self, path: Path):
        """Save summary as LaTeX table."""
        if self.summary is None:
            return

        # Use proposed metrics for the table
        proposed = [m for m in S.PROPOSED_METRICS
                    if m in self.summary["metric"].values]
        if not proposed:
            proposed = [m for m in S.METRICS_CLUSTERING
                        if m in self.summary["metric"].values]

        methods = self.method_order

        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{GAHIB benchmark results (mean $\\pm$ std across datasets)}")
        lines.append("\\label{tab:gahib_benchmark}")

        col_spec = "l" + "c" * len(proposed)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        header = "Method & " + " & ".join(
            S.METRIC_DISPLAY.get(m, m) for m in proposed
        ) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        for method in methods:
            cells = [S.get_display_name(method)]
            for metric in proposed:
                mask = (
                    (self.summary["method"] == method) &
                    (self.summary["metric"] == metric)
                )
                row = self.summary.loc[mask]
                if not row.empty:
                    mean_val = row["mean"].values[0]
                    std_val = row["std"].values[0]
                    if pd.notna(mean_val):
                        cells.append(f"${mean_val:.3f} \\pm {std_val:.3f}$")
                    else:
                        cells.append("--")
                else:
                    cells.append("--")
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _save_significance_table(self, path: Path, sig_pairs):
        """Save statistical significance results."""
        records = []
        proposed = [m for m in S.PROPOSED_METRICS
                    if m in self.long_data["metric"].values]
        if not proposed:
            proposed = [m for m in S.METRICS_CLUSTERING
                        if m in self.long_data["metric"].values]

        for ma, mb in sig_pairs:
            for metric in proposed:
                pval, stars = self.compute_significance(metric, ma, mb)
                records.append({
                    "method_a": ma,
                    "method_b": mb,
                    "metric": metric,
                    "p_value": pval,
                    "significance": stars,
                })

        pd.DataFrame(records).to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if not self._loaded:
            self.load_all()

    def _get_available_groups(self):
        groups = []
        for group_name, group_cfg in METRIC_GROUPS.items():
            available = [
                metric for metric in group_cfg["metrics"]
                if metric in self.long_data["metric"].values
            ]
            if not available:
                logger.debug("Skipping %s: no metrics available", group_name)
                continue
            groups.append((group_name, group_cfg, available))
        return groups

    @staticmethod
    def _normalize_method_list(methods):
        if methods is None:
            return None

        normalized = []
        seen = set()
        for method in methods:
            mapped = S.LEGACY_NAME_MAP.get(str(method), str(method))
            if mapped not in seen:
                normalized.append(mapped)
                seen.add(mapped)
        return normalized

    # Short display aliases: used on x-tick labels to avoid redundant
    # prefixes ("GM-VAE (Eucl.)" → "Eucl.") and long words ("Graph
    # Transformer" → "G-Trans") that otherwise clip the figure border.
    _LABEL_ABBREVIATIONS = {
        "GM-VAE (Eucl.)":   "Eucl.",
        "GM-VAE (Poinc.)":  "Poinc.",
        "GM-VAE (PGM)":     "PGM",
        "GM-VAE (L-PGM)":   "L-PGM",
        "GM-VAE (HW)":      "HW",
        "GraphTransformer": "G-Trans",
    }

    @classmethod
    def _format_method_label(cls, label: str) -> str:
        display = S.get_display_name(str(label))
        return cls._LABEL_ABBREVIATIONS.get(display, display)

    def _format_x_axis(self, ax, methods: List[str], font_scale: float = 1.0):
        labels = [self._format_method_label(method) for method in methods]

        # Keep rotation generous across the board so the all-metrics
        # row reads consistently — previously ablation sat at only 22°
        # and visually clashed with the other benchmark panels.
        max_len = max((len(lbl) for lbl in labels), default=0)
        if len(methods) <= 5 and max_len <= 6:
            rotation = 30
        elif len(methods) <= 8:
            rotation = 40
        else:
            rotation = 45
        pad = 0.5
        y_offset = -0.014 if rotation > 0 else -0.008
        ha = "right"

        fontsize = max(7, min(S.FS_TICK, int(84 / max(1, len(methods)))))
        fontsize = fontsize * font_scale
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        plt.setp(
            ax.get_xticklabels(),
            rotation=rotation,
            ha=ha,
            rotation_mode="anchor",
            fontsize=fontsize,
        )
        ax.tick_params(axis="x", pad=pad)
        for tick in ax.get_xticklabels():
            tick.set_y(y_offset)

    @staticmethod
    def _format_y_axis(ax, values):
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size == 0:
            return
        max_abs = np.max(np.abs(finite))
        min_abs = np.min(np.abs(finite[finite != 0])) if np.any(finite != 0) else max_abs
        if max_abs >= 1000 or (min_abs > 0 and min_abs < 0.01):
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 3))
            ax.yaxis.set_major_formatter(formatter)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            ax.yaxis.get_offset_text().set_size(S.FS_SMALL)

    def get_metric_summary(self, metric: str) -> pd.DataFrame:
        """Get mean/std for a single metric across all methods."""
        self._ensure_loaded()
        return self.summary[self.summary["metric"] == metric].copy()

    def get_available_metrics(self) -> List[str]:
        """Return list of all metrics found in the data."""
        self._ensure_loaded()
        return sorted(self.long_data["metric"].unique().tolist())

    def get_available_datasets(self) -> List[str]:
        """Return list of all loaded datasets."""
        return sorted(self.raw_data.keys())

    def __repr__(self):
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"VisualizationController(results_dir={str(self.results_dir)!r}, "
            f"status={status}, datasets={len(self.raw_data)}, "
            f"methods={self.method_names})"
        )
