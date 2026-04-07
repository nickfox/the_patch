#!/bin/bash
set -euo pipefail

source venv/bin/activate
mkdir -p logs

echo "=========================================="
echo "Step 1: Fit CCA (sweep K values)"
echo "=========================================="
python -u -m mlxmas.cca_adapter \
    --paired-data mlxmas/data/paired_states_S13_R22.npz \
    --sweep \
    2>&1 | tee logs/cca_fit.log

echo ""
echo "=========================================="
echo "Step 2: Diagnostics (K=512)"
echo "=========================================="
python -u -m mlxmas.diagnose_latents \
    --adapter-type cca \
    --adapter-path mlxmas/data/cca_adapter_S13_R22_K512.npz \
    --sender-layer 13 --receiver-layer 22 \
    2>&1 | tee logs/diag_cca_K512.log

echo ""
echo "=========================================="
echo "Step 3: GSM8K eval (5 questions, K=512)"
echo "=========================================="
python -u -m mlxmas.test_cross \
    --receiver mlx-community/gemma-2-9b-it-8bit \
    --adapter cca --adapter-path mlxmas/data/cca_adapter_S13_R22_K512.npz \
    --sender-layer 13 --start-layer 22 \
    --max-samples 5 \
    2>&1 | tee logs/eval_cca_K512.log

echo ""
echo "=========================================="
echo "Pipeline complete. Key logs:"
echo "  logs/cca_fit.log       — canonical correlations (THE diagnostic)"
echo "  logs/diag_cca_K512.log — logit decode & distribution"
echo "  logs/eval_cca_K512.log — GSM8K eval"
echo "=========================================="
