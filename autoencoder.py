"""Autoencoder baseline for self-supervised learning."""

from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils import EarlyStopping, log_progress


class AutoencoderModel(nn.Module):
    """Autoencoder model."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return self.decoder(encoded), encoded


def train_autoencoder(
    x_train, x_valid, parameters: Dict[str, int]
) -> nn.Module:
    """Train autoencoder with reconstruction loss."""
    _, dim = x_train.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoencoderModel(dim, parameters["hidden_dim"]).to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.001, alpha=0.9, eps=1e-7
    )
    mse_loss = nn.MSELoss()

    dataloader = DataLoader(
        TensorDataset(torch.from_numpy(x_train).float()),
        batch_size=parameters["batch_size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    x_valid_t = torch.from_numpy(x_valid).float().to(device)

    early_stopping = EarlyStopping(patience=parameters.get("patience", 5))

    for epoch in range(parameters["epochs"]):
        model.train()
        epoch_loss = 0.0
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)

            reconstructed, _ = model(batch_x)
            loss = mse_loss(reconstructed, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)

        model.eval()
        with torch.no_grad():
            val_reconstructed, _ = model(x_valid_t)
            val_loss = mse_loss(val_reconstructed, x_valid_t).item()

        log_progress(epoch, parameters["epochs"], {"train loss": epoch_loss}, val_loss)

        if early_stopping.step(val_loss, model):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    early_stopping.load_best(model)
    return model.encoder.eval()
