#!/bin/bash
set -euo pipefail

source venv/bin/activate
mkdir -p logs

echo "=========================================="
echo "Step 1: Collect 500 forward-pass pairs"
echo "=========================================="
python -u -m mlxmas.collect_paired_states \
    --sender mlx-community/Qwen3-4B-8bit \
    --receiver mlx-community/gemma-2-9b-it-8bit \
    --sender-layer 13 --receiver-layer 22 \
    --n-prompts 500 --n-positions 50 \
    --output mlxmas/data/paired_states_S13_R22.npz \
    2>&1 | tee logs/collect_clean.log

echo ""
echo "=========================================="
echo "Step 2: Fit CCA (sweep K values)"
echo "=========================================="
python -u -m mlxmas.cca_adapter \
    --paired-data mlxmas/data/paired_states_S13_R22.npz \
    --sweep \
    2>&1 | tee logs/cca_fit_clean.log

echo ""
echo "=========================================="
echo "Step 3: Train MLP adapter"
echo "=========================================="
python -u -m mlxmas.train_mlp_adapter \
    --paired-data mlxmas/data/paired_states_S13_R22.npz \
    --lr 1e-3 --batch-size 256 --epochs 200 \
    --output mlxmas/data/mlp_adapter_S13_R22.npz \
    2>&1 | tee logs/mlp_train_clean.log

echo ""
echo "=========================================="
echo "Step 4a: Diagnostics — CCA K=512"
echo "=========================================="
python -u -m mlxmas.diagnose_latents \
    --adapter-type cca \
    --adapter-path mlxmas/data/cca_adapter_S13_R22_K512.npz \
    --sender-layer 13 --receiver-layer 22 \
    2>&1 | tee logs/diag_cca_clean.log

echo ""
echo "=========================================="
echo "Step 4b: Diagnostics — MLP"
echo "=========================================="
python -u -m mlxmas.diagnose_latents \
    --adapter-type mlp \
    --adapter-path mlxmas/data/mlp_adapter_S13_R22.npz \
    --sender-layer 13 --receiver-layer 22 \
    2>&1 | tee logs/diag_mlp_clean.log

echo ""
echo "=========================================="
echo "Step 5a: GSM8K eval — CCA K=512 (5 questions)"
echo "=========================================="
python -u -m mlxmas.test_cross \
    --receiver mlx-community/gemma-2-9b-it-8bit \
    --adapter cca --adapter-path mlxmas/data/cca_adapter_S13_R22_K512.npz \
    --sender-layer 13 --start-layer 22 \
    --max-samples 5 \
    2>&1 | tee logs/eval_cca_clean.log

echo ""
echo "=========================================="
echo "Step 5b: GSM8K eval — MLP (5 questions)"
echo "=========================================="
python -u -m mlxmas.test_cross \
    --receiver mlx-community/gemma-2-9b-it-8bit \
    --adapter mlp --adapter-path mlxmas/data/mlp_adapter_S13_R22.npz \
    --sender-layer 13 --start-layer 22 \
    --max-samples 5 \
    2>&1 | tee logs/eval_mlp_clean.log

echo ""
echo "=========================================="
echo "Pipeline complete. Key logs:"
echo "  logs/collect_clean.log    — paired state collection"
echo "  logs/cca_fit_clean.log    — canonical correlations"
echo "  logs/mlp_train_clean.log  — MLP training curves"
echo "  logs/diag_cca_clean.log   — CCA diagnostics"
echo "  logs/diag_mlp_clean.log   — MLP diagnostics"
echo "  logs/eval_cca_clean.log   — CCA GSM8K eval"
echo "  logs/eval_mlp_clean.log   — MLP GSM8K eval"
echo "=========================================="
