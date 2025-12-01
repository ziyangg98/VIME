#!/bin/bash
set -e

# Default parameters
DATASET="synthetic"
N_NOISE=0
SEED=42
ITERATIONS=10

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --n_noise) N_NOISE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================================================"
echo "VIME Experiment"
echo "========================================================================"
echo "Dataset: $DATASET, Noise: $N_NOISE, Seed: $SEED, Iterations: $ITERATIONS"
echo "========================================================================"

mkdir -p results

# Run multiple iterations
for ((i=1; i<=ITERATIONS; i++)); do
    CURRENT_SEED=$((SEED + i - 1))
    echo ""
    echo "########################################################################"
    echo "# ITERATION $i/$ITERATIONS (seed=$CURRENT_SEED)"
    echo "########################################################################"

    # Generate data
    python data.py --dataset "$DATASET" --n_noise_features "$N_NOISE" --seed "$CURRENT_SEED"

    # Train models
    python baseline.py --seed "$CURRENT_SEED" --output_path "./results/baseline_$i.npy"
    python dae.py --seed "$CURRENT_SEED" --output_path "./results/dae_$i.npy"
    python vime.py --seed "$CURRENT_SEED" --output_path "./results/vime_$i.npy"

    echo "------------------------------------------------------------------------"
    python -c "
import numpy as np
print(f'Iteration $i: Baseline={np.load(\"./results/baseline_$i.npy\"):.4f}, DAE={np.load(\"./results/dae_$i.npy\"):.4f}, VIME={np.load(\"./results/vime_$i.npy\"):.4f}')
"
done

# Summary
echo ""
echo "========================================================================"
echo "FINAL RESULTS (Average over $ITERATIONS iterations)"
echo "========================================================================"
python -c "
import numpy as np
baseline = [np.load(f'./results/baseline_{i}.npy') for i in range(1, $ITERATIONS+1)]
dae = [np.load(f'./results/dae_{i}.npy') for i in range(1, $ITERATIONS+1)]
vime = [np.load(f'./results/vime_{i}.npy') for i in range(1, $ITERATIONS+1)]
print(f'Baseline: Avg={np.mean(baseline):.4f}, Std={np.std(baseline):.4f}')
print(f'DAE+MLP:  Avg={np.mean(dae):.4f}, Std={np.std(dae):.4f}')
print(f'VIME:     Avg={np.mean(vime):.4f}, Std={np.std(vime):.4f}')
"
echo "========================================================================"
