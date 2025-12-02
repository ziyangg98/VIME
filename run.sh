#!/bin/bash
set -e

# Default parameters
DATASET="synthetic"
N_NOISE=0
NOISE_LEVEL=0.1
SEED=42
ITERATIONS=10

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --n_noise) N_NOISE="$2"; shift 2 ;;
        --noise_level) NOISE_LEVEL="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================================================"
echo "VIME Experiment"
echo "========================================================================"
echo "Dataset: $DATASET, N_Noise: $N_NOISE, Noise_Level: $NOISE_LEVEL, Seed: $SEED, Iterations: $ITERATIONS"
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
    python data.py --dataset "$DATASET" --n_noise_features "$N_NOISE" --noise_level "$NOISE_LEVEL" --seed "$CURRENT_SEED"

    # Train models
    python baseline.py --seed "$CURRENT_SEED" --output_path "./results/baseline_$i.npy"
    python dae.py --seed "$CURRENT_SEED" --output_path "./results/dae_$i.npy"
    python vime.py --seed "$CURRENT_SEED"  --output_path "./results/vime_self_$i.npy"
    python vime.py --seed "$CURRENT_SEED" --semi --beta 0.1 --semi_patience 100 --output_path "./results/vime_semi_$i.npy"
    python adv.py --seed "$CURRENT_SEED" --output_path "./results/adv_$i.npy"

    echo "------------------------------------------------------------------------"
    python -c "
import numpy as np
print(f'Iteration $i: Baseline={np.load(\"./results/baseline_$i.npy\"):.4f}, DAE={np.load(\"./results/dae_$i.npy\"):.4f}, VIME-Self={np.load(\"./results/vime_self_$i.npy\"):.4f}, VIME-Semi={np.load(\"./results/vime_semi_$i.npy\"):.4f}, Adv={np.load(\"./results/adv_$i.npy\"):.4f}')
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
vime_self = [np.load(f'./results/vime_self_{i}.npy') for i in range(1, $ITERATIONS+1)]
vime_semi = [np.load(f'./results/vime_semi_{i}.npy') for i in range(1, $ITERATIONS+1)]
adv = [np.load(f'./results/adv_{i}.npy') for i in range(1, $ITERATIONS+1)]
print(f'Baseline:  Avg={np.mean(baseline):.4f}, Std={np.std(baseline):.4f}')
print(f'DAE+MLP:   Avg={np.mean(dae):.4f}, Std={np.std(dae):.4f}')
print(f'VIME-Self: Avg={np.mean(vime_self):.4f}, Std={np.std(vime_self):.4f}')
print(f'VIME-Semi: Avg={np.mean(vime_semi):.4f}, Std={np.std(vime_semi):.4f}')
print(f'Adv-SSL:   Avg={np.mean(adv):.4f}, Std={np.std(adv):.4f}')
"
echo "========================================================================"
