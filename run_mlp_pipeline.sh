#!/bin/bash
set -euo pipefail

source venv/bin/activate
mkdir -p logs

echo "=========================================="
echo "Step 1: Collect paired states (500 prompts)"
echo "=========================================="
python -u -m mlxmas.collect_paired_states \
    --sender mlx-community/Qwen3-4B-8bit \
    --receiver mlx-community/gemma-2-9b-it-8bit \
    --sender-layer 13 --receiver-layer 22 \
    --n-prompts 500 --n-positions 50 \
    --include-latent-states \
    --output mlxmas/data/paired_states_S13_R22.npz \
    2>&1 | tee logs/collect_paired_states.log

echo ""
echo "=========================================="
echo "Step 2: Train MLP adapter (cosine + MSE loss)"
echo "=========================================="
python -u -m mlxmas.train_mlp_adapter \
    --paired-data mlxmas/data/paired_states_S13_R22.npz \
    --lr 1e-3 --batch-size 256 --epochs 200 \
    --output mlxmas/data/mlp_adapter_S13_R22.npz \
    2>&1 | tee logs/train_mlp_adapter.log

echo ""
echo "=========================================="
echo "Step 3: Run diagnostics"
echo "=========================================="
python -u -m mlxmas.diagnose_latents \
    --adapter-type mlp \
    --adapter-path mlxmas/data/mlp_adapter_S13_R22.npz \
    --sender-layer 13 --receiver-layer 22 \
    2>&1 | tee logs/diag_mlp.log

echo ""
echo "=========================================="
echo "Step 4: GSM8K eval (5 questions)"
echo "=========================================="
python -u -m mlxmas.test_cross \
    --receiver mlx-community/gemma-2-9b-it-8bit \
    --adapter mlp --adapter-path mlxmas/data/mlp_adapter_S13_R22.npz \
    --sender-layer 13 --start-layer 22 \
    --max-samples 5 \
    2>&1 | tee logs/eval_mlp_5.log

echo ""
echo "=========================================="
echo "Pipeline complete. Check logs/"
echo "=========================================="
