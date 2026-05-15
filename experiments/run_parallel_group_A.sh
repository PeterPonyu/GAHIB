#!/usr/bin/env bash
set -euo pipefail
# Group A: Latent Dim Ablation + Computational Cost (sequential within group)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${GAHIB_PYTHON:-python}"

mkdir -p GAHIB_results

echo "[Group A] Starting: Latent Dim + Computational Cost"
echo "[Group A] Python: ${PYTHON_BIN}"
echo "[Group A] GPU device: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

"${PYTHON_BIN}" experiments/run_new_experiments_sequential.py --group A 2>&1

echo "[Group A] COMPLETE"
