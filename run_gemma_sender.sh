#!/bin/bash
# ===========================================================
# Gemma→Qwen reverse pipeline: calibration + evaluation
# ===========================================================
# Run from project root:
#   cd /Users/nickfox137/Documents/mlx_latentmas
#   ./run_gemma_sender.sh
#
# Memory: ~9GB (both 8-bit models loaded simultaneously)
# Time:   ~2-3 hours total
# ===========================================================

set -euo pipefail

cd "$(dirname "$0")"
source venv/bin/activate

MODELS_A="mlx-community/gemma-2-2b-it-8bit"
MODELS_B="mlx-community/Qwen3-4B-8bit"
LOGDIR="logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGDIR/gemma_sender_${TIMESTAMP}.log"

log() { echo "$(date '+%H:%M:%S') | $*" | tee -a "$LOGFILE"; }

log "=========================================="
log "Gemma→Qwen reverse pipeline"
log "=========================================="


# ----------------------------------------------------------
# Step 1: Qwen standalone baseline (no cross-model)
# ----------------------------------------------------------
log "STEP 1/6: Qwen standalone baseline (50 questions)"
python -u -m mlxmas.test_gemma_sender --qwen-only --max-samples 50 2>&1 | tee -a "$LOGFILE"
log "Step 1 complete."
echo

# ----------------------------------------------------------
# Step 2-5: Procrustes calibration (4 layer pairs)
# ----------------------------------------------------------
PAIRS=(
    "10 12"
    "10 9"
    "9  12"
    "12 14"
)

for i in "${!PAIRS[@]}"; do
    read -r SL RL <<< "${PAIRS[$i]}"
    STEP=$((i + 2))
    OUT="mlxmas/data/procrustes_gemma_S${SL}_qwen_R${RL}_multitoken.npz"

    if [ -f "$OUT" ]; then
        log "STEP ${STEP}/6: SKIP — $OUT already exists"
        continue
    fi

    log "STEP ${STEP}/6: Calibrate Gemma S${SL} → Qwen R${RL}"
    python -u -m mlxmas.contextual_procrustes \
        --multitoken \
        --model_a "$MODELS_A" \
        --model_b "$MODELS_B" \
        --sender_layer "$SL" \
        --receiver_layer "$RL" \
        --n_calibration 2000 \
        --n_positions 20 \
        --output "$OUT" 2>&1 | tee -a "$LOGFILE"
    log "Step ${STEP} complete: $OUT"
    echo
done

# ----------------------------------------------------------
# Step 6: Cross-model eval (priority 1 pair: S10→R12)
# ----------------------------------------------------------
log "STEP 6/6: Cross-model eval — Gemma S10 → Qwen R12 (50 questions)"
python -u -m mlxmas.test_gemma_sender \
    --sender-layer 10 \
    --receiver-layer 12 \
    --max-samples 50 2>&1 | tee -a "$LOGFILE"
log "Step 6 complete."

log "=========================================="
log "ALL DONE — full log: $LOGFILE"
log "=========================================="

