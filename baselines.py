"""Baseline supervised learning models."""

from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from utils import to_labels, to_one_hot, EarlyStopping, log_progress


def logit(x_train, y_train, x_test):
    """Logistic Regression."""
    y_train = to_labels(y_train) if len(y_train.shape) > 1 else y_train
    return (
        LogisticRegression(max_iter=1000)
        .fit(x_train, y_train)
        .predict_proba(x_test)
    )


def xgb_model(x_train, y_train, x_test):
    """XGBoost."""
    y_train = to_labels(y_train) if len(y_train.shape) > 1 else y_train
    return xgb.XGBClassifier().fit(x_train, y_train).predict_proba(x_test)


class MLPModel(nn.Module):
    """Multi-layer perceptron model."""

    def __init__(
        self,
        data_dim: int,
        label_dim: int,
        hidden_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        act_fn = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }.get(activation, nn.ReLU())
        self.network = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, label_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def mlp(x_train, y_train, x_valid, y_valid, x_test, parameters: Dict[str, int]):
    """Multi-layer perceptron (MLP)."""
    y_train = to_one_hot(y_train) if len(y_train.shape) == 1 else y_train
    y_valid = to_one_hot(y_valid) if len(y_valid.shape) == 1 else y_valid

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPModel(
        x_train.shape[1],
        y_train.shape[1],
        parameters["hidden_dim"],
        parameters["activation"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
        ),
        batch_size=parameters["batch_size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    y_valid_tensor = torch.from_numpy(y_valid).float().to(device)
    x_valid_tensor = torch.from_numpy(x_valid).float()

    early_stopping = EarlyStopping(patience=parameters.get("patience", 5))

    for epoch in range(parameters["epochs"]):
        model.train()
        epoch_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_valid_tensor.to(device))
            val_loss = F.cross_entropy(val_logits, y_valid_tensor).item()

        log_progress(epoch, parameters["epochs"], {"train loss": epoch_train_loss}, val_loss)

        if early_stopping.step(val_loss, model):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    early_stopping.load_best(model)

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test).float().to(device))
        return F.softmax(logits, dim=1).cpu().numpy()
