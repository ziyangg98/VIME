#!/bin/bash
set -e

DATASET="synthetic"
SEED=42
ITERATIONS=10

PM_VALUES=(0.1 0.3 0.5 0.7 0.9)
N_NOISE_VALUES=(0 50 100)

echo "========================================================================"
echo "VIME: pm vs n_noise"
echo "========================================================================"

mkdir -p results

RESULTS_FILE="results/pm_noise_results.csv"
echo "pm,n_noise,vime_self_avg,vime_self_std" > "$RESULTS_FILE"

for PM in "${PM_VALUES[@]}"; do
    for N_NOISE in "${N_NOISE_VALUES[@]}"; do
        echo ""
        echo "Testing pm=$PM, n_noise=$N_NOISE"

        RESULT_DIR="results/pm${PM}_noise${N_NOISE}"
        mkdir -p "$RESULT_DIR"

        for ((i=1; i<=ITERATIONS; i++)); do
            CURRENT_SEED=$((SEED + i - 1))
            echo "  Iteration $i/$ITERATIONS (seed=$CURRENT_SEED)"

            python data.py --dataset "$DATASET" --n_noise_features "$N_NOISE" --seed "$CURRENT_SEED"
            python vime.py --seed "$CURRENT_SEED" --p_m "$PM" --output_path "$RESULT_DIR/vime_self_$i.npy"
        done

        python -c "
import numpy as np
result_dir = '$RESULT_DIR'
iterations = $ITERATIONS
pm = $PM
n_noise = $N_NOISE

vime_self = [np.load(f'{result_dir}/vime_self_{i}.npy') for i in range(1, iterations+1)]

print(f'pm={pm}, n_noise={n_noise}: {np.mean(vime_self):.4f} ± {np.std(vime_self):.4f}')

with open('$RESULTS_FILE', 'a') as f:
    f.write(f'{pm},{n_noise},{np.mean(vime_self):.4f},{np.std(vime_self):.4f}\n')
"
    done
done

echo ""
echo "========================================================================"
echo "RESULTS"
echo "========================================================================"
cat "$RESULTS_FILE"
