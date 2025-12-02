"""Adv-SSL for tabular self-supervised learning."""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from baseline import train_supervised
from utils import (
    EarlyStopping,
    corrupt_samples,
    generate_mask,
    load_data,
    log_progress,
    set_seed,
    transform_with_encoder,
)


def adv_loss(z0: torch.Tensor, z1: torch.Tensor, G, lambda_param: float = 5e-2):
    """Adv-SSL loss function.

    Args:
        z0: First view embeddings (N, D)
        z1: Second view embeddings (N, D)
        G: Gradient matrix for adversarial update (None for first pair)
        lambda_param: Weight for cross-correlation term
    """
    N, D = z0.shape

    # L2 normalize and scale by sqrt(D)
    z0 = np.sqrt(D) * F.normalize(z0, dim=1)
    z1 = np.sqrt(D) * F.normalize(z1, dim=1)

    # Cross-correlation matrix
    c = z0.T @ z1 / N  # (D, D)
    c_diff = c - torch.eye(D, device=c.device)

    # Adversarial gradient: use previous c_diff as G
    if G is None:
        G = c_diff

    # Alignment loss (L2 distance between positive pairs)
    align_loss = (z0 - z1).norm(p=2, dim=1).pow(2).mean()

    # Uniformity regularization via Frobenius inner product
    uniform_loss = torch.trace(c_diff.T @ G)

    return align_loss + lambda_param * uniform_loss, c_diff.detach()


class AdvEncoder(nn.Module):
    """Adv-SSL encoder."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def create_augmented_views(X: np.ndarray, p_mask: float, num_views: int = 2):
    """Create multiple augmented views via corruption."""
    views = []
    for _ in range(num_views):
        mask = generate_mask(p_mask, X)
        _, X_aug = corrupt_samples(mask, X)
        views.append(X_aug)
    return views


def train_adv(
    X_train,
    X_val,
    p_mask: float,
    lambda_param: float,
    params: Dict[str, int],
) -> nn.Module:
    """Train Adv-SSL encoder for tabular data."""
    _, dim = X_train.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_views = params.get("num_views", 2)

    model = AdvEncoder(dim, params["hidden_dim"]).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-7)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float()),
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    X_val_t = torch.from_numpy(X_val).float()

    early_stopping = EarlyStopping(patience=params.get("patience", 5))
    num_pairs = num_views * (num_views - 1) // 2

    for epoch in range(params["epochs"]):
        model.train()
        total_loss, n_batches = 0.0, 0

        for (X_batch,) in loader:
            X_np = X_batch.numpy()

            # Create multiple augmented views
            views = create_augmented_views(X_np, p_mask, num_views)
            views_t = [torch.from_numpy(v).float().to(device) for v in views]

            # Forward all views through model
            embeddings = [model(v) for v in views_t]

            # Compute loss over all pairs
            loss = 0
            G = None
            for i in range(num_views - 1):
                for j in range(i + 1, num_views):
                    pair_loss, G = adv_loss(embeddings[i], embeddings[j], G, lambda_param)
                    loss = loss + pair_loss
            loss = loss / num_pairs

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validation loss
        model.eval()
        with torch.no_grad():
            X_val_np = X_val_t.numpy()
            views_val = create_augmented_views(X_val_np, p_mask, num_views)
            views_val_t = [torch.from_numpy(v).float().to(device) for v in views_val]
            emb_val = [model(v) for v in views_val_t]

            val_loss = 0
            G = None
            for i in range(num_views - 1):
                for j in range(i + 1, num_views):
                    pair_loss, G = adv_loss(emb_val[i], emb_val[j], G, lambda_param)
                    val_loss = val_loss + pair_loss
            val_loss = (val_loss / num_pairs).item()

        log_progress(epoch, params["epochs"], {"train": total_loss / n_batches}, val_loss)

        if early_stopping.step(val_loss, model):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    early_stopping.load_best(model)
    return model.encoder.eval()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data_cache/data.npz")
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--p_m", type=float, default=0.3)
    parser.add_argument("--lambda_param", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="./results/adv.npy")
    args = parser.parse_args()

    set_seed(args.seed)
    data = load_data(args.data_path)

    # Train Adv-SSL encoder
    print("Training Adv-SSL encoder...")
    X_all = np.vstack([data["X_train"], data["X_unlabeled"]])
    encoder = train_adv(
        X_all,
        data["X_val"],
        args.p_m,
        args.lambda_param,
        {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "hidden_dim": data["X_train"].shape[1],
            "num_views": args.num_views,
            "patience": args.patience,
        },
    )

    # Evaluate with MLP
    print("Evaluating Adv-SSL + MLP...")
    X_train_enc, X_val_enc, X_test_enc = transform_with_encoder(
        encoder, data["X_train"], data["X_val"], data["X_test"]
    )
    acc = train_supervised(
        X_train_enc,
        data["y_train"],
        X_val_enc,
        data["y_val"],
        X_test_enc,
        data["y_test"],
        "mlp",
        "acc",
        args.hidden_dim,
        args.patience,
    )

    print(f"Adv-SSL + MLP Accuracy: {acc:.4f}")
    np.save(args.output_path, acc)
