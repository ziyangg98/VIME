"""VIME self-supervised learning."""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils import generate_mask, corrupt_samples, EarlyStopping, log_progress


class VIMESelfModel(nn.Module):
    """VIME self-supervised model."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU()
        )
        self.mask_estimator = nn.Sequential(
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )
        self.feature_estimator = nn.Sequential(
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return self.mask_estimator(encoded), self.feature_estimator(encoded)


def vime_self(
    X_train, X_val, p_mask: float, alpha: float, params: Dict[str, int]
) -> nn.Module:
    """VIME self-supervised learning."""
    _, dim = X_train.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VIMESelfModel(dim, params["hidden_dim"]).to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.001, alpha=0.9, eps=1e-7
    )

    # Generate corrupted samples
    mask_train, mask_val = generate_mask(p_mask, X_train), generate_mask(
        p_mask, X_val
    )
    mask_label, X_corrupt = corrupt_samples(mask_train, X_train)
    mask_label_val, X_corrupt_val = corrupt_samples(mask_val, X_val)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_corrupt).float(),
            torch.from_numpy(mask_label).float(),
            torch.from_numpy(X_train).float(),
        ),
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    # Validation tensors
    X_corrupt_val_t = torch.from_numpy(X_corrupt_val).float().to(device)
    mask_label_val_t = torch.from_numpy(mask_label_val).float().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)

    early_stopping = EarlyStopping(patience=params.get("patience", 5))

    for epoch in range(params["epochs"]):
        model.train()
        total_loss, mask_loss_sum, feat_loss_sum, n_batches = 0.0, 0.0, 0.0, 0
        for X_corrupt_b, mask_label_b, X_orig_b in loader:
            X_corrupt_b = X_corrupt_b.to(device, non_blocking=True)
            mask_label_b = mask_label_b.to(device, non_blocking=True)
            X_orig_b = X_orig_b.to(device, non_blocking=True)

            mask_pred, feat_pred = model(X_corrupt_b)
            mask_loss = F.binary_cross_entropy(mask_pred, mask_label_b)
            feat_loss = F.mse_loss(feat_pred, X_orig_b)
            loss = mask_loss + alpha * feat_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            mask_loss_sum += mask_loss.item()
            feat_loss_sum += feat_loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            mask_pred_val, feat_pred_val = model(X_corrupt_val_t)
            val_loss = (
                F.binary_cross_entropy(mask_pred_val, mask_label_val_t).item()
                + alpha * F.mse_loss(feat_pred_val, X_val_t).item()
            )

        log_progress(
            epoch,
            params["epochs"],
            {
                "train": total_loss / n_batches,
                "mask": mask_loss_sum / n_batches,
                "feat": feat_loss_sum / n_batches,
            },
            val_loss,
        )

        if early_stopping.step(val_loss, model):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    early_stopping.load_best(model)
    return model.encoder.eval()
