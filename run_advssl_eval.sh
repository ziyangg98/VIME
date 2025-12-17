#!/bin/bash
set -e

SEED=42
ITERATIONS=10

echo "========================================================================"
echo "AdvSSL Evaluation with Tuned Parameters (10 iterations each)"
echo "========================================================================"

mkdir -p results/advssl_eval

# n_noise=0, best params: p_m=0.409, lambda=0.0106, hidden_dim=200, num_views=3, batch_size=256, patience=9
echo ""
echo ">>> n_noise=0, noise_level=0.1"
for ((i=1; i<=ITERATIONS; i++)); do
    CURRENT_SEED=$((SEED + i - 1))
    echo "  Iteration $i/$ITERATIONS (seed=$CURRENT_SEED)"
    python data.py --dataset synthetic --n_noise_features 0 --noise_level 0.1 --seed "$CURRENT_SEED"
    python adv.py --seed "$CURRENT_SEED" \
        --p_m 0.409 --lambda_param 0.0106 --hidden_dim 200 --num_views 3 --batch_size 256 --patience 9 \
        --output_path "results/advssl_eval/n0_$i.npy"
done

# n_noise=50, best params: p_m=0.440, lambda=0.0165, hidden_dim=200, num_views=4, batch_size=256, patience=8
echo ""
echo ">>> n_noise=50, noise_level=0.1"
for ((i=1; i<=ITERATIONS; i++)); do
    CURRENT_SEED=$((SEED + i - 1))
    echo "  Iteration $i/$ITERATIONS (seed=$CURRENT_SEED)"
    python data.py --dataset synthetic --n_noise_features 50 --noise_level 0.1 --seed "$CURRENT_SEED"
    python adv.py --seed "$CURRENT_SEED" \
        --p_m 0.440 --lambda_param 0.0165 --hidden_dim 200 --num_views 4 --batch_size 256 --patience 8 \
        --output_path "results/advssl_eval/n50_$i.npy"
done

# n_noise=100, best params: p_m=0.497, lambda=0.0641, hidden_dim=50, num_views=2, batch_size=256, patience=12
echo ""
echo ">>> n_noise=100, noise_level=0.1"
for ((i=1; i<=ITERATIONS; i++)); do
    CURRENT_SEED=$((SEED + i - 1))
    echo "  Iteration $i/$ITERATIONS (seed=$CURRENT_SEED)"
    python data.py --dataset synthetic --n_noise_features 100 --noise_level 0.1 --seed "$CURRENT_SEED"
    python adv.py --seed "$CURRENT_SEED" \
        --p_m 0.497 --lambda_param 0.0641 --hidden_dim 50 --num_views 2 --batch_size 256 --patience 12 \
        --output_path "results/advssl_eval/n100_$i.npy"
done

# Compute statistics
echo ""
echo "========================================================================"
echo "RESULTS SUMMARY"
echo "========================================================================"
python -c "
import numpy as np

for n_noise in [0, 50, 100]:
    accs = [np.load(f'results/advssl_eval/n{n_noise}_{i}.npy') for i in range(1, 11)]
    print(f'n_noise={n_noise}: {np.mean(accs):.4f} +/- {np.std(accs):.4f} (min={np.min(accs):.4f}, max={np.max(accs):.4f})')
"
