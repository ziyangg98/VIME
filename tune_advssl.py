"""AdvSSL hyperparameter tuning with Optuna (parallel)."""

import argparse
import os
import numpy as np
import optuna
from optuna.samplers import TPESampler
from concurrent.futures import ProcessPoolExecutor, as_completed

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_single_eval(params, seed, trial_id, eval_id):
    """Run a single evaluation."""
    from utils import set_seed, load_data, transform_with_encoder
    from adv import train_adv
    from baseline import train_supervised

    set_seed(seed)
    data = load_data("./data_cache/data.npz")

    X_all = np.vstack([data["X_train"], data["X_unlabeled"]])
    encoder = train_adv(
        X_all,
        data["X_val"],
        params["p_m"],
        params["lambda_param"],
        {
            "batch_size": params["batch_size"],
            "epochs": 1000,
            "hidden_dim": data["X_train"].shape[1],
            "num_views": params["num_views"],
            "patience": params["patience"],
        },
    )

    X_train_enc, X_val_enc, X_test_enc = transform_with_encoder(
        encoder, data["X_train"], data["X_val"], data["X_test"]
    )
    acc = train_supervised(
        X_train_enc, data["y_train"],
        X_val_enc, data["y_val"],
        X_test_enc, data["y_test"],
        "mlp", "acc", params["hidden_dim"], params["patience"],
    )
    return acc


def objective(trial, seed, n_evals, executor):
    """Optuna objective function with parallel evals."""
    params = {
        "p_m": trial.suggest_float("p_m", 0.1, 0.5),
        "lambda_param": trial.suggest_float("lambda_param", 0.005, 0.3, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [50, 100, 200]),
        "num_views": trial.suggest_int("num_views", 2, 4),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "patience": trial.suggest_int("patience", 5, 20),
    }

    # Run evaluations in parallel
    futures = [
        executor.submit(run_single_eval, params, seed + i, trial.number, i)
        for i in range(n_evals)
    ]
    accs = [f.result() for f in as_completed(futures)]
    return np.mean(accs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--n_noise", type=int, default=0)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_evals", type=int, default=3, help="Evaluations per config")
    parser.add_argument("--n_jobs", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    os.makedirs("results/advssl_tuning", exist_ok=True)

    # Generate data first
    print(f"Generating data: n_noise={args.n_noise}, noise_level={args.noise_level}")
    import subprocess
    subprocess.run([
        "python", "data.py",
        "--dataset", "synthetic",
        "--n_noise_features", str(args.n_noise),
        "--noise_level", str(args.noise_level),
        "--seed", str(args.seed),
    ], check=True)

    # Run optimization with parallel executor
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
    )

    print(f"Starting optimization with {args.n_jobs} parallel workers...")
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        study.optimize(
            lambda trial: objective(trial, args.seed, args.n_evals, executor),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

    # Print results
    print("\n" + "=" * 60)
    print("BEST PARAMETERS")
    print("=" * 60)
    print(f"Best accuracy: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save results
    df = study.trials_dataframe()
    df.to_csv(f"results/advssl_tuning/optuna_n{args.n_noise}_l{args.noise_level}.csv", index=False)
    print(f"\nResults saved to results/advssl_tuning/")


if __name__ == "__main__":
    main()
