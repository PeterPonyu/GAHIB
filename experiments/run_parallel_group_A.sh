#!/bin/bash
# Group A: Latent Dim Ablation + Computational Cost (sequential within group)
cd /home/zeyufu/Desktop/GAHIB
DL_PY=/home/zeyufu/.conda/envs/dl/bin/python

echo "[Group A] Starting: Latent Dim + Computational Cost"
echo "[Group A] GPU device: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

$DL_PY experiments/run_new_experiments_sequential.py --group A 2>&1

echo "[Group A] COMPLETE"
