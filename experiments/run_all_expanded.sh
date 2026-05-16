#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${GAHIB_PYTHON:-python}"

# Safety gate: this entrypoint launches the expensive full 53-dataset benchmark.
# It must not start unless local data roots are complete and the runtime has
# been explicitly acknowledged.  On failure, the Python guard prints the exact
# public-safe run plan instead of executing any experiment.
"${PYTHON_BIN}" -m experiments.benchmark_config

LOG_PATH="GAHIB_results/run_expanded.log"
mkdir -p GAHIB_results

run_study() {
    local label="$1"
    local script="$2"
    echo ""
    echo "${label}"
    "${PYTHON_BIN}" "${script}" 2>&1 | tee -a "${LOG_PATH}"
}

echo "============================================================"
echo "GAHIB Expanded Benchmark: 53 datasets, 11 experiments"
echo "============================================================"
echo "Python: ${PYTHON_BIN}"
echo "Dataset dirs: ${GAHIB_DATASET_DIRS:-not set}"

run_study "[1/11] Ablation Study (5 variants)" experiments/run_ablation.py
run_study "[2/11] SC Deep Learning Benchmark (8 methods)" experiments/run_sc_deeplearning_benchmark.py
run_study "[3/11] Classical DR Benchmark (6 methods)" experiments/run_classical_benchmark.py
run_study "[4/11] GM-VAE Benchmark (6 methods)" experiments/run_gmvae_benchmark.py
run_study "[5/11] Disentanglement Comparison (6 methods)" experiments/run_disentanglement.py
run_study "[6/11] Encoder Architecture Comparison (3 methods)" experiments/run_encoder_comparison.py
run_study "[7/11] Graph Convolution Sweep (6 methods)" experiments/run_graph_conv_sweep.py
run_study "[8/11] Hyperparameter Sensitivity (4 sweeps x 5 values)" experiments/run_hyperparam_sensitivity.py
run_study "[9/11] Latent Dimension Ablation (5 dimensions)" experiments/run_latent_dim_ablation.py
run_study "[10/11] Multi-Seed Robustness (5 seeds)" experiments/run_seed_robustness.py
run_study "[11/11] Computational Cost Analysis (3 methods + scaling)" experiments/run_computational_cost.py

echo ""
echo "============================================================"
echo "ALL 11 EXPERIMENTS COMPLETE"
echo "============================================================"

for exp in ablation sc_deeplearning_benchmark classical_benchmark gmvae_benchmark disentanglement encoder_comparison graph_conv_sweep hyperparam_sensitivity latent_dim_ablation seed_robustness computational_cost; do
    n=$(find "GAHIB_results/${exp}/tables" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')
    echo "  ${exp}: ${n} datasets"
done
