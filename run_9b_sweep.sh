#!/bin/bash
# Contextual Procrustes layer sweep for Gemma-2-9B receiver
# Finds the best (sender_layer=13, receiver_layer) pair, then runs end-to-end test
set -e

cd /Users/nickfox137/Documents/mlx_latentmas
source venv/bin/activate

RECEIVER="mlx-community/gemma-2-9b-it-8bit"
SENDER_LAYER=13
LAYERS=(16 18 20)
BEST_COS=-1
BEST_LAYER=-1

echo "=== Gemma-2-9B Contextual Procrustes Sweep ==="
echo "Sender layer: $SENDER_LAYER"
echo "Receiver layers: ${LAYERS[*]}"
echo ""

for RL in "${LAYERS[@]}"; do
    OUT="mlxmas/data/procrustes_9b_S${SENDER_LAYER}_R${RL}.npz"
    echo "================================================"
    echo "Calibrating S${SENDER_LAYER} -> R${RL}..."
    echo "================================================"

    python -u -m mlxmas.contextual_procrustes \
        --model_b "$RECEIVER" \
        --sender_layer "$SENDER_LAYER" \
        --receiver_layer "$RL" \
        --multitoken --n_positions 20 \
        --n_calibration 500 --heldout_frac 0.2 \
        --output "$OUT"

    # Extract held-out cosine from the output file
    COS=$(python -c "
import numpy as np
d = np.load('$OUT')
print(f\"{float(d['heldout_cosine']):.4f}\")
")
    echo "  -> S${SENDER_LAYER}->R${RL} held-out cosine: $COS"

    # Track best
    IS_BETTER=$(python -c "print(1 if $COS > $BEST_COS else 0)")
    if [ "$IS_BETTER" = "1" ]; then
        BEST_COS=$COS
        BEST_LAYER=$RL
    fi
    echo ""
done

echo "================================================"
echo "SWEEP RESULTS"
echo "================================================"
echo "Best pair: S${SENDER_LAYER} -> R${BEST_LAYER} (cosine: $BEST_COS)"
echo ""

# Run end-to-end test with the best layer pair
BEST_NPZ="mlxmas/data/procrustes_9b_S${SENDER_LAYER}_R${BEST_LAYER}.npz"
echo "================================================"
echo "Running end-to-end GSM8K test (5 questions)"
echo "  Procrustes: $BEST_NPZ"
echo "  Inject at Gemma layer $BEST_LAYER"
echo "================================================"

python -u -m mlxmas.test_cross \
    --receiver "$RECEIVER" \
    --procrustes "$BEST_NPZ" \
    --sender-layer "$SENDER_LAYER" \
    --start-layer "$BEST_LAYER" \
    --max-samples 5 \
    --latent-steps 10

echo ""
echo "Done."
