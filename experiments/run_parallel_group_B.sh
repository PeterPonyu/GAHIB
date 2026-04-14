#!/bin/bash
# Group B: Seed Robustness + Hyperparameter Sensitivity (sequential within group)
cd /home/zeyufu/Desktop/GAHIB
DL_PY=/home/zeyufu/.conda/envs/dl/bin/python

echo "[Group B] Starting: Seed Robustness + HP Sensitivity"
echo "[Group B] GPU device: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

$DL_PY experiments/run_new_experiments_sequential.py --group B 2>&1

echo "[Group B] COMPLETE"
