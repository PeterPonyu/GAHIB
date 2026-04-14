#!/bin/bash
set -e
cd /home/zeyufu/Desktop/GAHIB

echo "============================================================"
echo "GAHIB Expanded Benchmark: 53 datasets, 11 experiments"
echo "============================================================"

echo ""
echo "[1/11] Ablation Study (5 variants)"
python3 experiments/run_ablation.py 2>&1 | tee -a GAHIB_results/run_expanded.log

echo ""
echo "[2/11] SC Deep Learning Benchmark (8 methods)"
python3 experiments/run_sc_deeplearning_benchmark.py 2>&1 | tee -a GAHIB_results/run_expanded.log

echo ""
echo "[3/11] Classical DR Benchmark (6 methods)"
python3 experiments/run_classical_benchmark.py 2>&1 | tee -a GAHIB_results/run_expanded.log

echo ""
echo "[4/11] GM-VAE Benchmark (6 methods)"
python3 experiments/run_gmvae_benchmark.py 2>&1 | tee -a GAHIB_results/run_expanded.log

echo ""
echo "[5/11] Disentanglement Comparison (6 methods)"
python3 experiments/run_disentanglement.py 2>&1 | tee -a GAHIB_results/run_expanded.log

echo ""
echo "[6/11] Encoder Architecture Comparison (3 methods)"
python3 experiments/run_encoder_comparison.py 2>&1 | tee -a GAHIB_results/run_expanded.log

echo ""
echo "[7/11] Graph Convolution Sweep (6 methods)"
python3 experiments/run_graph_conv_sweep.py 2>&1 | tee -a GAHIB_results/run_expanded.log

echo ""
echo "[8/11] Hyperparameter Sensitivity (4 sweeps x 5 values)"
python3 experiments/run_hyperparam_sensitivity.py 2>&1 | tee -a GAHIB_results/run_expanded.log

echo ""
echo "[9/11] Latent Dimension Ablation (5 dimensions)"
python3 experiments/run_latent_dim_ablation.py 2>&1 | tee -a GAHIB_results/run_expanded.log

echo ""
echo "[10/11] Multi-Seed Robustness (5 seeds)"
python3 experiments/run_seed_robustness.py 2>&1 | tee -a GAHIB_results/run_expanded.log

echo ""
echo "[11/11] Computational Cost Analysis (3 methods + scaling)"
python3 experiments/run_computational_cost.py 2>&1 | tee -a GAHIB_results/run_expanded.log

echo ""
echo "============================================================"
echo "ALL 11 EXPERIMENTS COMPLETE"
echo "============================================================"

# Count results
for exp in ablation sc_deeplearning_benchmark classical_benchmark gmvae_benchmark disentanglement encoder_comparison graph_conv_sweep hyperparam_sensitivity latent_dim_ablation seed_robustness computational_cost; do
    n=$(ls GAHIB_results/$exp/tables/*.csv 2>/dev/null | wc -l)
    echo "  $exp: $n datasets"
done
