# GAHIB Paper Asset Index

`paper/gahib_paper.tex` currently uses 19 figures and 8 tables.
The files under `paper/figures/` expose the generated assets used by LaTeX;
the canonical outputs live under `GAHIB_results/`.

## Figures

| No. | Label | Paper asset | Generated target | Generator |
| --- | --- | --- | --- | --- |
| 1 | `fig:overview` | `overview.pdf` | `GAHIB_results/figures/fig_overview.pdf` | `gahib/viz/fig_overview.py` |
| 2 | `fig:architecture` | `architecture.pdf` | `GAHIB_results/figures/fig_architecture.pdf` | `gahib/viz/fig_architecture.py` |
| 3 | `fig:taxonomy` | `dataset_taxonomy.pdf` | `GAHIB_results/figures/fig_dataset_taxonomy.pdf` | `gahib/viz/fig_dataset_taxonomy.py` |
| 4 | `fig:ablation` | `ablation_all_metrics.pdf` | `GAHIB_results/ablation/figures/fig_all_metrics.pdf` | `gahib/viz/controller.py` via `python -m gahib.viz.run_all_visualizations` |
| 5 | `fig:scdeep` | `scdeep_all_metrics.pdf` | `GAHIB_results/sc_deeplearning_benchmark/figures/fig_all_metrics.pdf` | `gahib/viz/controller.py` via `python -m gahib.viz.run_all_visualizations` |
| 6 | `fig:classical` | `classical_all_metrics.pdf` | `GAHIB_results/classical_benchmark/figures/fig_all_metrics.pdf` | `gahib/viz/controller.py` via `python -m gahib.viz.run_all_visualizations` |
| 7 | `fig:gmvae` | `gmvae_all_metrics.pdf` | `GAHIB_results/gmvae_benchmark/figures/fig_all_metrics.pdf` | `gahib/viz/controller.py` via `python -m gahib.viz.run_all_visualizations` |
| 8 | `fig:disent` | `disent_all_metrics.pdf` | `GAHIB_results/disentanglement/figures/fig_all_metrics.pdf` | `gahib/viz/controller.py` via `python -m gahib.viz.run_all_visualizations` |
| 9 | `fig:encoder` | `encoder_all_metrics.pdf` | `GAHIB_results/encoder_comparison/figures/fig_all_metrics.pdf` | `gahib/viz/controller.py` via `python -m gahib.viz.run_all_visualizations` |
| 10 | `fig:gconv` | `gconv_all_metrics.pdf` | `GAHIB_results/graph_conv_sweep/figures/fig_all_metrics.pdf` | `gahib/viz/controller.py` via `python -m gahib.viz.run_all_visualizations` |
| 11 | `fig:interp_embedding` | `interp_embedding_poincare.pdf` | `GAHIB_results/interpretation/figures/fig_themed_embedding_poincare.pdf` | `gahib/viz/interpretation.py` |
| 12 | `fig:interp_bottleneck` | `interp_bottleneck.pdf` | `GAHIB_results/interpretation/figures/fig_themed_bottleneck.pdf` | `gahib/viz/interpretation.py` |
| 13 | `fig:interp_hyperbolic` | `interp_hyperbolic.pdf` | `GAHIB_results/interpretation/figures/fig_themed_hyperbolic.pdf` | `gahib/viz/interpretation.py` |
| 14 | `fig:downstream` | `fig_downstream_analysis.pdf` | `GAHIB_results/figures/fig_downstream_analysis.pdf` | `gahib/viz/fig_downstream_analysis.py` |
| 15 | `fig:interp_summary` | `interp_summary.pdf` | `GAHIB_results/interpretation/figures/fig_themed_gene_stemness.pdf` | `gahib/viz/interpretation.py` |
| 16 | `fig:latdim` | `latent_dim_ablation.pdf` | `GAHIB_results/latent_dim_ablation/figures/fig_latent_dim_ablation.pdf` | `gahib/viz/fig_new_experiments.py` via `python -m gahib.viz.run_all_visualizations` |
| 17 | `fig:seeds` | `seed_robustness.pdf` | `GAHIB_results/seed_robustness/figures/fig_seed_robustness.pdf` | `gahib/viz/fig_new_experiments.py` via `python -m gahib.viz.run_all_visualizations` |
| 18 | `fig:cost` | `computational_cost.pdf` | `GAHIB_results/computational_cost/figures/fig_computational_cost.pdf` | `gahib/viz/fig_new_experiments.py` via `python -m gahib.viz.run_all_visualizations` |
| 19 | `fig:hpsens` | `hyperparam_sensitivity.pdf` | `GAHIB_results/hyperparam_sensitivity/figures/fig_hyperparam_sensitivity.pdf` | `gahib/viz/fig_new_experiments.py` via `python -m gahib.viz.run_all_visualizations` |

## Tables

| No. | Label | Content | Source |
| --- | --- | --- | --- |
| 1 | `tab:ablation` | Component ablation summary | `GAHIB_results/ablation/figures/statistical_summary.csv` |
| 2 | `tab:scdeep` | Deep-learning benchmark (8 paper-selected methods) | `GAHIB_results/sc_deeplearning_benchmark/figures/statistical_summary.csv` |
| 3 | `tab:classical` | Classical dimensionality-reduction benchmark | `GAHIB_results/classical_benchmark/figures/statistical_summary.csv` |
| 4 | `tab:gmvae` | Geometric VAE benchmark | `GAHIB_results/gmvae_benchmark/figures/statistical_summary.csv` |
| 5 | `tab:disent` | Disentanglement comparison | `GAHIB_results/disentanglement/figures/statistical_summary.csv` |
| 6 | `tab:encoder` | Encoder comparison | `GAHIB_results/encoder_comparison/figures/statistical_summary.csv` |
| 7 | `tab:gconv` | Graph convolution sweep | `GAHIB_results/graph_conv_sweep/figures/statistical_summary.csv` |
| 8 | `tab:interp` | Interpretation summary | `GAHIB_results/interpretation/tables/interp_summary.csv` |

## Regeneration

```bash
make figures
make
```

Equivalent figure commands from the repository root:

```bash
python -m gahib.viz.fig_overview
python -m gahib.viz.fig_architecture
python -m gahib.viz.fig_dataset_taxonomy
python -m gahib.viz.run_all_visualizations
python -m gahib.viz.fig_downstream_analysis
python experiments/run_interpretation.py --figures-only
```
