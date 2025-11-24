"""VIME training script."""

import argparse
import numpy as np
import os
import torch
import random

from data import load_mnist_data, load_synthetic_data
from baselines import logit, xgb_model, mlp
from self_supervised import vime_self as train_self_supervised
from semi_supervised import vime_semi as train_semi_supervised
from autoencoder import train_autoencoder
from utils import evaluate

# Print separators
SEP_MAJOR = "=" * 70
SEP_MINOR = "-" * 70
SEP_HEADER = "#" * 70


def transform_with_encoder(encoder, X_train, X_val, X_test):
    """Transform data using a trained encoder."""
    device = next(encoder.parameters()).device
    with torch.no_grad():
        X_train_enc = (
            encoder(torch.from_numpy(X_train).float().to(device)).cpu().numpy()
        )
        X_val_enc = (
            encoder(torch.from_numpy(X_val).float().to(device)).cpu().numpy()
        )
        X_test_enc = (
            encoder(torch.from_numpy(X_test).float().to(device)).cpu().numpy()
        )
    return X_train_enc, X_val_enc, X_test_enc


def set_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_supervised(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    model,
    metric,
    hidden_dim=100,
    patience=50,
):
    """Train supervised models."""
    if model in ("logit", "xgboost"):
        X_full = np.vstack([X_train, X_val])
        y_full = np.vstack([y_train, y_val])
        y_pred = (
            logit(X_full, y_full, X_test)
            if model == "logit"
            else xgb_model(X_full, y_full, X_test)
        )
    elif model == "mlp":
        params = {
            "hidden_dim": hidden_dim,
            "epochs": 1000,
            "activation": "relu",
            "batch_size": 100,
            "patience": patience,
        }
        y_pred = mlp(X_train, y_train, X_val, y_val, X_test, params)
    else:
        raise ValueError(f"Unknown model: {model}")

    return evaluate(metric, y_test, y_pred)


def train_vime(
    label_rate,
    models,
    n_labeled,
    p_mask,
    alpha,
    K,
    beta,
    dataset="mnist",
    n_noise=0,
    hidden_dim=100,
    patience=50,
):
    """VIME main training pipeline."""
    results = np.zeros(len(models) + 3)

    # Load data
    if dataset == "mnist":
        X_train, y_train, X_unlabeled, X_val, y_val, X_test, y_test = load_mnist_data(
            label_rate
        )
    else:
        X_train, y_train, X_unlabeled, X_val, y_val, X_test, y_test = (
            load_synthetic_data(
                label_rate, total_features=200, n_noise_features=n_noise
            )
        )

    X_train, y_train = X_train[:n_labeled], y_train[:n_labeled]
    input_dim = X_train.shape[1]

    # Train supervised models
    print(f"\n{SEP_MAJOR}\nSTEP 1: Training Supervised Baseline ({models[0].upper()})\n{SEP_MAJOR}")
    for i, model in enumerate(models):
        results[i] = train_supervised(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            model,
            "acc",
            hidden_dim,
            patience,
        )
    print(f"Supervised baseline accuracy: {results[0]:.4f}")

    # Train Autoencoder
    print(f"\n{SEP_MAJOR}\nSTEP 2: Training Autoencoder\n{SEP_MAJOR}")
    X_all = np.vstack([X_train, X_unlabeled])
    encoder_params = {
        "batch_size": 128,
        "epochs": 1000,
        "hidden_dim": input_dim,
        "patience": patience,
    }
    encoder_ae = train_autoencoder(X_all, X_val, encoder_params)

    os.makedirs("save_model", exist_ok=True)
    torch.save(encoder_ae, "./save_model/autoencoder_encoder.pth")

    # Test Autoencoder
    print(f"\n{SEP_MINOR}\nTesting Autoencoder + MLP classifier\n{SEP_MINOR}")
    X_train_ae, X_val_ae, X_test_ae = transform_with_encoder(
        encoder_ae, X_train, X_val, X_test
    )

    results[len(models)] = train_supervised(
        X_train_ae,
        y_train,
        X_val_ae,
        y_val,
        X_test_ae,
        y_test,
        "mlp",
        "acc",
        hidden_dim,
        patience,
    )
    print(f"Autoencoder + MLP accuracy: {results[len(models)]:.4f}")

    # Train VIME-Self
    print(f"\n{SEP_MAJOR}\nSTEP 3: Training VIME-Self\n{SEP_MAJOR}")
    encoder_self = train_self_supervised(
        X_all, X_val, p_mask, alpha, encoder_params
    )

    encoder_path = "./save_model/encoder_model.pth"
    torch.save(encoder_self, encoder_path)

    # Test VIME-Self
    print(f"\n{SEP_MINOR}\nTesting VIME-Self + MLP classifier\n{SEP_MINOR}")
    X_train_self, X_val_self, X_test_self = transform_with_encoder(
        encoder_self, X_train, X_val, X_test
    )

    results[len(models) + 1] = train_supervised(
        X_train_self,
        y_train,
        X_val_self,
        y_val,
        X_test_self,
        y_test,
        "mlp",
        "acc",
        hidden_dim,
        patience,
    )
    print(f"VIME-Self + MLP accuracy: {results[len(models)+1]:.4f}")

    # Train VIME-Semi
    print(f"\n{SEP_MAJOR}\nSTEP 4: Training VIME-Semi (End-to-End)\n{SEP_MAJOR}")
    semi_params = {
        "hidden_dim": hidden_dim,
        "batch_size": 128,
        "iterations": 1000,
        "patience": 100,
    }
    y_pred = train_semi_supervised(
        X_train,
        y_train,
        X_val,
        y_val,
        X_unlabeled,
        X_test,
        semi_params,
        p_mask,
        K,
        beta,
        encoder_path,
    )

    results[len(models) + 2] = evaluate("acc", y_test, y_pred)
    print(f"VIME-Semi accuracy: {results[len(models)+2]:.4f}")
    return results


def run_experiments(args):
    """Run experiments with multiple iterations."""
    results = np.zeros([args.iterations, 4])

    for i in range(args.iterations):
        set_seed(args.seed + i)
        print(f"\n{SEP_HEADER}\n# ITERATION {i+1}/{args.iterations}\n{SEP_HEADER}")

        results[i, :] = train_vime(
            args.label_data_rate,
            [args.model_name],
            args.label_no,
            args.p_m,
            args.alpha,
            args.K,
            args.beta,
            args.dataset,
            args.n_noise_features,
            args.hidden_dim,
            args.patience,
        )

        print(f"\n{SEP_MINOR}")
        print(
            f"Iteration {i+1} Results: Supervised={results[i,0]:.4f}, "
            f"Autoencoder={results[i,1]:.4f}, VIME-Self={results[i,2]:.4f}, VIME={results[i,3]:.4f}"
        )
        print(SEP_MINOR)

    # Print final results
    print(f"\n{SEP_HEADER}\n# FINAL RESULTS (Average over {args.iterations} iterations)\n{SEP_HEADER}")
    for i, name in enumerate(["Supervised", "Autoencoder", "VIME-Self", "VIME"]):
        print(
            f"{name}: Avg={np.mean(results[:, i]):.4f}, Std={np.std(results[:, i]):.4f}"
        )
    print(SEP_HEADER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--model_name", choices=["logit", "xgboost", "mlp"], default="mlp"
    )
    parser.add_argument("--label_no", type=int, default=1000)
    parser.add_argument("--p_m", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--label_data_rate", type=float, default=0.1)
    parser.add_argument(
        "--dataset", choices=["mnist", "synthetic"], default="synthetic"
    )
    parser.add_argument("--n_noise_features", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)

    run_experiments(parser.parse_args())
