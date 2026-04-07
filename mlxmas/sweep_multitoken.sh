#!/bin/bash
# Sweep all 8 layer pairs from the injection probe test
# with multi-token Procrustes calibration (2000 prompts × 20 positions)
#
# Run: cd /Users/nickfox137/Documents/mlx_latentmas && bash mlxmas/sweep_multitoken.sh

set -e
cd /Users/nickfox137/Documents/mlx_latentmas
source venv/bin/activate

PAIRS=(
    "13,12"
    "13,10"
    "11,10"
    "11,9"
    "7,8"
    "13,9"
    "11,6"
    "7,4"
)

echo "=== Multi-token Procrustes sweep: ${#PAIRS[@]} pairs ==="
echo "  Started: $(date)"
echo ""

for pair in "${PAIRS[@]}"; do
    IFS=',' read -r sender receiver <<< "$pair"
    outfile="mlxmas/data/procrustes_S${sender}_R${receiver}_multitoken.npz"

    # Skip if already computed
    if [ -f "$outfile" ]; then
        echo "--- Skipping S${sender}→R${receiver} (already exists) ---"
        echo ""
        continue
    fi

    echo "=== Pair: Qwen L${sender} → Gemma L${receiver} ==="
    echo "  Time: $(date)"

    python -m mlxmas.contextual_procrustes \
        --multitoken \
        --sender_layer "$sender" \
        --receiver_layer "$receiver" \
        --n_calibration 2000 \
        --n_positions 20 \
        --output "$outfile"

    echo ""
    echo "  Completed: $(date)"
    echo ""
done

echo "=== All pairs complete ==="
echo "  Finished: $(date)"
echo ""

# Summary: extract held-out cosine from each .npz
echo "=== Results Summary ==="
python3 -c "
import numpy as np, glob, os
files = sorted(glob.glob('mlxmas/data/procrustes_S*_R*_multitoken.npz'))
header = '{:<20} {:>10} {:>12} {:>8}'.format('Pair', 'Train Cos', 'Heldout Cos', 'Gap')
print(header)
print('-' * 52)
for f in files:
    d = np.load(f)
    name = os.path.basename(f).replace('procrustes_','').replace('_multitoken.npz','')
    tc = float(d.get('train_cosine', 0))
    hc = float(d.get('heldout_cosine', 0))
    row = '{:<20} {:>10.4f} {:>12.4f} {:>8.4f}'.format(name, tc, hc, tc-hc)
    print(row)
"
