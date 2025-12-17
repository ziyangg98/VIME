#!/bin/bash
set -e

DATASET="synthetic"
SEED=42
ITERATIONS=10

N_NOISE_VALUES=(0 50 100)
NOISE_LEVEL_VALUES=(0.1)

echo "========================================================================"
echo "VIME: n_noise vs noise_level"
echo "========================================================================"

mkdir -p results

RESULTS_FILE="results/noise_level_results.csv"
echo "n_noise,noise_level,baseline_avg,baseline_std,dae_avg,dae_std,vime_self_avg,vime_self_std,vime_semi_avg,vime_semi_std,adv_avg,adv_std" > "$RESULTS_FILE"

for N_NOISE in "${N_NOISE_VALUES[@]}"; do
    for NOISE_LEVEL in "${NOISE_LEVEL_VALUES[@]}"; do
        echo ""
        echo "Testing n_noise=$N_NOISE, noise_level=$NOISE_LEVEL"

        RESULT_DIR="results/noise${N_NOISE}_level${NOISE_LEVEL}"
        mkdir -p "$RESULT_DIR"

        for ((i=1; i<=ITERATIONS; i++)); do
            CURRENT_SEED=$((SEED + i - 1))
            echo "  Iteration $i/$ITERATIONS (seed=$CURRENT_SEED)"

            python data.py --dataset "$DATASET" --n_noise_features "$N_NOISE" --noise_level "$NOISE_LEVEL" --seed "$CURRENT_SEED"
            python baseline.py --seed "$CURRENT_SEED" --output_path "$RESULT_DIR/baseline_$i.npy"
            python dae.py --seed "$CURRENT_SEED" --output_path "$RESULT_DIR/dae_$i.npy"
            python vime.py --seed "$CURRENT_SEED" --output_path "$RESULT_DIR/vime_self_$i.npy"
            python vime.py --seed "$CURRENT_SEED" --semi --beta 0.1 --semi_patience 100 --output_path "$RESULT_DIR/vime_semi_$i.npy"
            python adv.py --seed "$CURRENT_SEED" --output_path "$RESULT_DIR/adv_$i.npy"
        done

        python -c "
import numpy as np
result_dir = '$RESULT_DIR'
iterations = $ITERATIONS
n_noise = $N_NOISE
noise_level = $NOISE_LEVEL

baseline = [np.load(f'{result_dir}/baseline_{i}.npy') for i in range(1, iterations+1)]
dae = [np.load(f'{result_dir}/dae_{i}.npy') for i in range(1, iterations+1)]
vime_self = [np.load(f'{result_dir}/vime_self_{i}.npy') for i in range(1, iterations+1)]
vime_semi = [np.load(f'{result_dir}/vime_semi_{i}.npy') for i in range(1, iterations+1)]
adv = [np.load(f'{result_dir}/adv_{i}.npy') for i in range(1, iterations+1)]

print(f'n_noise={n_noise}, noise_level={noise_level}:')
print(f'  Baseline:  {np.mean(baseline):.4f} +/- {np.std(baseline):.4f}')
print(f'  DAE:       {np.mean(dae):.4f} +/- {np.std(dae):.4f}')
print(f'  VIME-Self: {np.mean(vime_self):.4f} +/- {np.std(vime_self):.4f}')
print(f'  VIME-Semi: {np.mean(vime_semi):.4f} +/- {np.std(vime_semi):.4f}')
print(f'  Adv-SSL:   {np.mean(adv):.4f} +/- {np.std(adv):.4f}')

with open('$RESULTS_FILE', 'a') as f:
    f.write(f'{n_noise},{noise_level},{np.mean(baseline):.4f},{np.std(baseline):.4f},{np.mean(dae):.4f},{np.std(dae):.4f},{np.mean(vime_self):.4f},{np.std(vime_self):.4f},{np.mean(vime_semi):.4f},{np.std(vime_semi):.4f},{np.mean(adv):.4f},{np.std(adv):.4f}\n')
"
    done
done

echo ""
echo "========================================================================"
echo "RESULTS SAVED TO: $RESULTS_FILE"
echo "========================================================================"
cat "$RESULTS_FILE"