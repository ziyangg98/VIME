"""Data loading utilities for MNIST and synthetic data."""

import numpy as np
import pandas as pd
from torchvision import datasets


def split_labeled_unlabeled(X_train, y_train, label_data_rate, seed=None):
    """Split training data into labeled and unlabeled subsets."""
    if seed is not None:
        np.random.seed(seed)
    idx = np.random.permutation(len(y_train))
    split_idx = int(len(idx) * label_data_rate)
    X_label, y_label = X_train[idx[:split_idx]], y_train[idx[:split_idx]]
    X_unlab = X_train[idx[split_idx:]]
    return X_label, y_label, X_unlab


def load_mnist_data(label_data_rate, seed=42):
    """Load MNIST data with train/valid/test split."""
    train_dataset = datasets.MNIST(root="./data", train=True, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True)

    X_train_all = train_dataset.data.numpy()
    y_train_all = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    # One-hot encoding and normalization
    y_train_all = np.asarray(pd.get_dummies(y_train_all))
    y_test = np.asarray(pd.get_dummies(y_test))
    X_train_all = X_train_all.reshape(len(X_train_all), -1) / 255.0
    X_test = X_test.reshape(len(X_test), -1) / 255.0

    # Split into train (80%) and valid (20%)
    n_train = int(len(X_train_all) * 0.8)
    X_train, y_train = X_train_all[:n_train], y_train_all[:n_train]
    X_valid, y_valid = X_train_all[n_train:], y_train_all[n_train:]

    # Split training data into labeled and unlabeled
    X_label, y_label, X_unlab = split_labeled_unlabeled(
        X_train, y_train, label_data_rate, seed=seed
    )

    return X_label, y_label, X_unlab, X_valid, y_valid, X_test, y_test


def generate_synthetic_data(
    n_samples=10000,
    total_features=200,
    n_latent=10,
    n_classes=10,
    noise_level=0.1,
    n_noise_features=0,
    seed=None,
):
    """Generate synthetic data from latent variables."""
    if seed is not None:
        np.random.seed(seed)

    n_informative = total_features - n_noise_features
    z = np.random.randn(n_samples, n_latent)
    W_x = np.random.randn(n_latent, n_informative) * 2
    W_y = np.random.randn(n_latent, n_classes) * 2

    # Generate informative features with min-max normalization
    X_informative = z @ W_x + np.random.randn(n_samples, n_informative) * noise_level
    X_min, X_max = X_informative.min(axis=0, keepdims=True), X_informative.max(
        axis=0, keepdims=True
    )
    X_informative = (X_informative - X_min) / (X_max - X_min + 1e-8)

    # Generate labels with softmax
    y_logits = z @ W_y + np.random.randn(n_samples, n_classes) * noise_level
    y_probs = np.exp(y_logits - y_logits.max(axis=1, keepdims=True))
    y_probs /= y_probs.sum(axis=1, keepdims=True)
    y_labels = np.array([np.random.choice(n_classes, p=p) for p in y_probs])
    y = np.eye(n_classes)[y_labels]

    # Add noise features if needed
    X = (
        np.hstack([X_informative, np.random.rand(n_samples, n_noise_features)])
        if n_noise_features > 0
        else X_informative
    )

    return X, y, z


def load_synthetic_data(
    label_data_rate=0.1,
    n_samples=10000,
    total_features=200,
    n_latent=10,
    n_classes=10,
    noise_level=0.1,
    n_noise_features=0,
    seed=42,
):
    """Load synthetic data with train/valid/test split."""
    X_all, y_all, _ = generate_synthetic_data(
        n_samples,
        total_features,
        n_latent,
        n_classes,
        noise_level,
        n_noise_features,
        seed,
    )

    # Split: train (64%), valid (16%), test (20%)
    n_train, n_valid = int(n_samples * 0.64), int(n_samples * 0.16)
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_valid, y_valid = (
        X_all[n_train : n_train + n_valid],
        y_all[n_train : n_train + n_valid],
    )
    X_test, y_test = X_all[n_train + n_valid :], y_all[n_train + n_valid :]

    # Split training into labeled/unlabeled
    X_label, y_label, X_unlab = split_labeled_unlabeled(
        X_train, y_train, label_data_rate, seed=seed + 1 if seed else None
    )

    return X_label, y_label, X_unlab, X_valid, y_valid, X_test, y_test


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["mnist", "synthetic"], default="synthetic"
    )
    parser.add_argument("--label_data_rate", type=float, default=0.1)
    parser.add_argument("--label_no", type=int, default=1000)
    parser.add_argument("--total_features", type=int, default=200)
    parser.add_argument("--n_noise_features", type=int, default=0)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./data_cache")
    args = parser.parse_args()

    if args.dataset == "mnist":
        X_train, y_train, X_unlabeled, X_val, y_val, X_test, y_test = load_mnist_data(
            args.label_data_rate, seed=args.seed
        )
    else:
        X_train, y_train, X_unlabeled, X_val, y_val, X_test, y_test = (
            load_synthetic_data(
                args.label_data_rate,
                total_features=args.total_features,
                noise_level=args.noise_level,
                n_noise_features=args.n_noise_features,
                seed=args.seed,
            )
        )

    X_train, y_train = X_train[: args.label_no], y_train[: args.label_no]

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        os.path.join(args.output_dir, "data.npz"),
        X_train=X_train,
        y_train=y_train,
        X_unlabeled=X_unlabeled,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    print(f"Data saved: X_train={X_train.shape}, X_unlabeled={X_unlabeled.shape}")
