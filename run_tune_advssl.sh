#!/bin/bash
set -e

echo "========================================================================"
echo "AdvSSL Hyperparameter Tuning (Optuna Bayesian Optimization)"
echo "========================================================================"

# Install optuna if needed
pip install optuna -q

# Run tuning for each noise config
for N_NOISE in 0 50 100; do
    echo ""
    echo ">>> Tuning for n_noise=$N_NOISE, noise_level=0.1"
    python tune_advssl.py --n_noise $N_NOISE --noise_level 0.1 --n_trials 50 --n_evals 3
done

echo ""
echo "========================================================================"
echo "TUNING COMPLETE - Results in results/advssl_tuning/"
echo "========================================================================"
