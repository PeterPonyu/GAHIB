#!/usr/bin/env bash
set -euo pipefail
# Group B: Seed Robustness + Hyperparameter Sensitivity (sequential within group)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${GAHIB_PYTHON:-python}"

mkdir -p GAHIB_results

echo "[Group B] Starting: Seed Robustness + HP Sensitivity"
echo "[Group B] Python: ${PYTHON_BIN}"
echo "[Group B] GPU device: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

"${PYTHON_BIN}" experiments/run_new_experiments_sequential.py --group B 2>&1

echo "[Group B] COMPLETE"
