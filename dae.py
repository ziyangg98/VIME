"""Denoising Autoencoder (DAE) for self-supervised learning."""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline import train_supervised
from utils import (
    EarlyStopping,
    load_data,
    log_progress,
    set_seed,
    transform_with_encoder,
)


class Autoencoder(nn.Module):
    """Autoencoder model."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return self.decoder(encoded), encoded


def train_dae(x_train, x_valid, params: Dict[str, int]) -> nn.Module:
    """Train autoencoder with reconstruction loss."""
    _, dim = x_train.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(dim, params["hidden_dim"]).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-7)
    mse_loss = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train).float()),
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    x_valid_t = torch.from_numpy(x_valid).float().to(device)
    early_stopping = EarlyStopping(patience=params.get("patience", 5))

    for epoch in range(params["epochs"]):
        model.train()
        epoch_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            reconstructed, _ = model(batch_x)
            loss = mse_loss(reconstructed, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)

        model.eval()
        with torch.no_grad():
            val_loss = mse_loss(model(x_valid_t)[0], x_valid_t).item()

        log_progress(epoch, params["epochs"], {"train loss": epoch_loss}, val_loss)

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
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="./results/dae.npy")
    args = parser.parse_args()

    set_seed(args.seed)
    data = load_data(args.data_path)

    # Train DAE
    print("Training DAE...")
    X_all = np.vstack([data["X_train"], data["X_unlabeled"]])
    encoder = train_dae(
        X_all,
        data["X_val"],
        {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "hidden_dim": data["X_train"].shape[1],
            "patience": args.patience,
        },
    )

    # Evaluate with MLP
    print("Evaluating DAE + MLP...")
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

    print(f"DAE + MLP Accuracy: {acc:.4f}")
    np.save(args.output_path, acc)
