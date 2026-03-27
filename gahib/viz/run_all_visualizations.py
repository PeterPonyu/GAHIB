#!/usr/bin/env python3
"""
GAHIB: Run All Visualizations
==============================

Fully automatic script that discovers all GAHIB benchmark results and
generates publication-quality figures for every experiment series.

Covers:
  1. Ablation study (Base VAE -> VAE+IB -> VAE+Hyp -> VAE+IB+Hyp -> GAHIB)
  2. GM-VAE geometric benchmark (5 external GM-VAE + GAHIB)
  3. Disentanglement regularization (VAE, beta-VAE, DIP, TC, Info, GAHIB)
  4. Cross-dataset benchmark (MLP, GAT, GAT+IB, GAT+IB+Lor, etc.)

For each series, generates:
  - Clustering boxplots (NMI, ARI, ASW, DAV, CAL, COR)
  - DRE UMAP series (distance_correlation, Q_local, Q_global, overall)
  - DRE t-SNE series
  - LSE intrinsic series (7 metrics)
  - DREX extended DR series (7 metrics)
  - LSEX extended latent series (5 metrics)
  - Summary heatmap
  - Mean +/- std tables (CSV + LaTeX)
  - Statistical significance tables

Usage:
    python -m gahib.viz.run_all_visualizations
"""

import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gahib.viz.controller import VisualizationController
from gahib.viz import style as S


def _is_gat_family_method(method_name: str) -> bool:
    name = str(method_name).lower()
    return name.startswith("gat") or ("gat" in name) or ("gahib" in name)


def _prioritize_gat_on_right(methods):
    """Reorder methods so GAT-family variants are plotted on the right side."""
    non_gat = [m for m in methods if not _is_gat_family_method(m)]
    gat = [m for m in methods if _is_gat_family_method(m) and "gahib" not in str(m).lower()]
    gahib = [m for m in methods if "gahib" in str(m).lower()]
    ordered = non_gat + gat + gahib
    return ordered if ordered else list(methods)


def _build_sig_pairs(methods):
    """Compare all baselines against a GAT-family focal method when available."""
    ordered = list(methods)
    focal = None
    for candidate in reversed(ordered):
        if _is_gat_family_method(candidate):
            focal = candidate
            break
    if focal is None and ordered:
        focal = ordered[-1]
    return [(m, focal) for m in ordered if m != focal]


def run_experiment_visualization(
    tables_dir, output_dir, method_names, method_order=None,
    palette=None, experiment_name="Experiment"
):
    """Run visualization for one experiment series."""
    tables_path = Path(tables_dir)
    if not tables_path.exists():
        print(f"  Skipping {experiment_name}: {tables_path} not found")
        return None

    csv_count = len(list(tables_path.glob("*.csv")))
    if csv_count == 0:
        print(f"  Skipping {experiment_name}: no CSV files in {tables_path}")
        return None

    print(f"\n{'='*70}")
    print(f"  {experiment_name}")
    print(f"  Tables dir: {tables_dir}")
    print(f"  Methods: {method_names}")
    print(f"  Datasets: {csv_count} CSV files")
    print(f"{'='*70}")

    ctrl = VisualizationController(
        results_dir=tables_dir,
        method_names=method_names,
        method_order=method_order or method_names,
        palette=palette,
    )
    ctrl.load_all()

    # Order methods by performance gap with GAHIB (worst→best, GAHIB rightmost)
    focal = None
    for candidate in reversed(ctrl.method_order):
        if "gahib" in str(candidate).lower():
            focal = candidate
            break
    if focal:
        ctrl.sort_methods_by_performance_gap(focal_method=focal)
    else:
        ctrl.sort_methods_by_performance_gap(focal_method=ctrl.method_order[-1])

    sig_pairs = _build_sig_pairs(ctrl.method_order)

    print(f"\n  Available metrics: {len(ctrl.get_available_metrics())}")
    print(f"  Available datasets: {ctrl.get_available_datasets()}")
    print(f"  Method order (perf-sorted): {ctrl.method_order}")

    results = ctrl.generate_all_figures(
        output_dir=output_dir,
        sig_pairs=sig_pairs,
    )

    print(f"\n  Generated {len(results)} outputs for {experiment_name}")
    return ctrl


def main():
    results_base = PROJECT_ROOT / "GAHIB_results"

    print(f"\n{'#'*70}")
    print(f"  GAHIB: Automatic Visualization System")
    print(f"  Project: {PROJECT_ROOT}")
    print(f"{'#'*70}")

    # ══════════════════════════════════════════════
    # Experiment 1: Ablation (5 methods)
    # ══════════════════════════════════════════════
    ablation_methods = ["Base VAE", "VAE+IB", "VAE+Hyp", "VAE+IB+Hyp", "GAHIB"]
    ablation_palette = dict(zip(ablation_methods, [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"
    ]))
    run_experiment_visualization(
        tables_dir=results_base / "ablation" / "tables",
        output_dir=results_base / "ablation" / "figures",
        method_names=ablation_methods,
        palette=ablation_palette,
        experiment_name="Ablation Study",
    )

    # ══════════════════════════════════════════════
    # Experiment 2: GM-VAE Benchmark (6 methods)
    # ══════════════════════════════════════════════
    gmvae_methods = [
        "GM-VAE (Eucl.)", "GM-VAE (Poinc.)", "GM-VAE (PGM)",
        "GM-VAE (L-PGM)", "GM-VAE (HW)", "GAHIB"
    ]
    gmvae_palette = dict(zip(gmvae_methods, [
        "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00"
    ]))
    run_experiment_visualization(
        tables_dir=results_base / "gmvae_benchmark" / "tables",
        output_dir=results_base / "gmvae_benchmark" / "figures",
        method_names=gmvae_methods,
        palette=gmvae_palette,
        experiment_name="GM-VAE Geometric Benchmark",
    )

    # ══════════════════════════════════════════════
    # Experiment 3: Disentanglement (6 methods)
    # ══════════════════════════════════════════════
    disent_methods = ["VAE", "beta-VAE", "DIP-VAE", "TC-VAE", "InfoVAE", "GAHIB"]
    disent_palette = dict(zip(disent_methods, [
        "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00"
    ]))
    run_experiment_visualization(
        tables_dir=results_base / "disentanglement" / "tables",
        output_dir=results_base / "disentanglement" / "figures",
        method_names=disent_methods,
        palette=disent_palette,
        experiment_name="Disentanglement Regularization",
    )

    # ══════════════════════════════════════════════
    # Experiment 6: Graph Conv Sweep (6 methods)
    # ══════════════════════════════════════════════
    gconv_methods = ["GCN", "GraphSAGE", "GAT", "Cheb", "TAG", "GraphTransformer"]
    gconv_palette = dict(zip(gconv_methods, [
        "#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73", "#CC79A7"
    ]))
    run_experiment_visualization(
        tables_dir=results_base / "graph_conv_sweep" / "tables",
        output_dir=results_base / "graph_conv_sweep" / "figures",
        method_names=gconv_methods,
        palette=gconv_palette,
        experiment_name="Graph Convolution Type Sweep",
    )

    # ══════════════════════════════════════════════
    # Experiment 7: Encoder Comparison (3 methods)
    # ══════════════════════════════════════════════
    enc_methods = ["MLP", "GAT", "Transformer"]
    enc_palette = dict(zip(enc_methods, [
        "#0072B2", "#D55E00", "#009E73"
    ]))
    run_experiment_visualization(
        tables_dir=results_base / "encoder_comparison" / "tables",
        output_dir=results_base / "encoder_comparison" / "figures",
        method_names=enc_methods,
        palette=enc_palette,
        experiment_name="Encoder Type Comparison",
    )

    # ══════════════════════════════════════════════
    # Experiment 8: SC Deep Learning Benchmark (8 methods)
    # ══════════════════════════════════════════════
    # Removed scSMD, scDAC, scGCC, siVAE — degenerate or misleading profiles
    sc_dl_methods = [
        "scVI", "CellBLAST", "CLEAR", "SCALEX",
        "scDeepCluster", "scDHMap", "scGNN", "GAHIB"
    ]
    sc_dl_palette = dict(zip(sc_dl_methods, [
        "#0072B2", "#56B4E9", "#009E73", "#F0E442",
        "#CC79A7", "#D55E00", "#999999", "#AA3377"
    ]))
    run_experiment_visualization(
        tables_dir=results_base / "sc_deeplearning_benchmark" / "tables",
        output_dir=results_base / "sc_deeplearning_benchmark" / "figures",
        method_names=sc_dl_methods,
        palette=sc_dl_palette,
        experiment_name="SC Deep Learning Benchmark",
    )

    # ══════════════════════════════════════════════
    # Experiment 9: Classical Benchmark (6 methods)
    # ══════════════════════════════════════════════
    classical_methods = ["PCA", "ICA", "NMF", "TruncatedSVD", "DiffusionMap", "GAHIB"]
    classical_palette = dict(zip(classical_methods, [
        "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00"
    ]))
    run_experiment_visualization(
        tables_dir=results_base / "classical_benchmark" / "tables",
        output_dir=results_base / "classical_benchmark" / "figures",
        method_names=classical_methods,
        palette=classical_palette,
        experiment_name="Classical DR Benchmark",
    )

    # ══════════════════════════════════════════════
    # Experiment 10: Interpretation & Biovalidation
    # ══════════════════════════════════════════════
    # This experiment has its own figure pipeline (not boxplot-based).
    # We check if results exist and report status.
    interp_dir = results_base / "interpretation"
    interp_figures = interp_dir / "figures"
    interp_tables = interp_dir / "tables"
    if interp_tables.exists() and any(interp_tables.glob("interp_*_bioval.json")):
        n_datasets = len(list(interp_tables.glob("interp_*_bioval.json")))
        n_figures = sum(1 for _ in interp_figures.rglob("*.pdf")) if interp_figures.exists() else 0
        print(f"\n{'='*70}")
        print(f"  Interpretation & Biovalidation")
        print(f"  {n_datasets} datasets completed, {n_figures} figure files")
        print(f"  (Run experiments/run_interpretation.py to generate/update)")
        print(f"{'='*70}")
    else:
        print(f"\n  Skipping Interpretation: run experiments/run_interpretation.py first")

    print(f"\n{'#'*70}")
    print(f"  ALL VISUALIZATIONS COMPLETE")
    print(f"{'#'*70}")

    # Print output locations
    for d in [results_base / "ablation" / "figures",
              results_base / "gmvae_benchmark" / "figures",
              results_base / "disentanglement" / "figures",
              results_base / "graph_conv_sweep" / "figures",
              results_base / "encoder_comparison" / "figures",
              results_base / "sc_deeplearning_benchmark" / "figures",
              results_base / "classical_benchmark" / "figures",
              results_base / "interpretation" / "figures"]:
        if d.exists():
            n_files = len(list(d.glob("*")))
            print(f"  {d}: {n_files} files")


if __name__ == "__main__":
    main()
