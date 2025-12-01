"""Baseline supervised learning models."""

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from utils import MLP, EarlyStopping, evaluate, log_progress, to_labels, to_one_hot


def logit(x_train, y_train, x_test):
    """Logistic Regression."""
    y_train = to_labels(y_train) if len(y_train.shape) > 1 else y_train
    return LogisticRegression(max_iter=1000).fit(x_train, y_train).predict_proba(x_test)


def xgb_model(x_train, y_train, x_test):
    """XGBoost."""
    y_train = to_labels(y_train) if len(y_train.shape) > 1 else y_train
    return xgb.XGBClassifier().fit(x_train, y_train).predict_proba(x_test)


def mlp(x_train, y_train, x_valid, y_valid, x_test, params: Dict[str, int]):
    """Multi-layer perceptron (MLP)."""
    y_train = to_one_hot(y_train) if len(y_train.shape) == 1 else y_train
    y_valid = to_one_hot(y_valid) if len(y_valid.shape) == 1 else y_valid

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(
        x_train.shape[1],
        y_train.shape[1],
        params["hidden_dim"],
        params["activation"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
        ),
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    y_valid_t = torch.from_numpy(y_valid).float().to(device)
    x_valid_t = torch.from_numpy(x_valid).float()
    early_stopping = EarlyStopping(patience=params.get("patience", 5))

    for epoch in range(params["epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            loss = F.cross_entropy(model(batch_x), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)

        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(x_valid_t.to(device)), y_valid_t).item()

        log_progress(epoch, params["epochs"], {"train loss": epoch_loss}, val_loss)

        if early_stopping.step(val_loss, model):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    early_stopping.load_best(model)
    model.eval()
    with torch.no_grad():
        return (
            F.softmax(model(torch.from_numpy(x_test).float().to(device)), dim=1)
            .cpu()
            .numpy()
        )


def train_supervised(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    model_name,
    metric,
    hidden_dim=100,
    patience=50,
):
    """Train supervised models."""
    if model_name in ("logit", "xgboost"):
        X_full = np.vstack([X_train, X_val])
        y_full = np.vstack([y_train, y_val])
        y_pred = (
            logit(X_full, y_full, X_test)
            if model_name == "logit"
            else xgb_model(X_full, y_full, X_test)
        )
    elif model_name == "mlp":
        y_pred = mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            {
                "hidden_dim": hidden_dim,
                "epochs": 1000,
                "activation": "relu",
                "batch_size": 100,
                "patience": patience,
            },
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return evaluate(metric, y_test, y_pred)


if __name__ == "__main__":
    import argparse

    from utils import load_data, set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data_cache/data.npz")
    parser.add_argument("--model", choices=["logit", "xgboost", "mlp"], default="mlp")
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="./results/baseline.npy")
    args = parser.parse_args()

    set_seed(args.seed)
    data = load_data(args.data_path)

    print(f"Training {args.model.upper()} baseline...")
    acc = train_supervised(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
        args.model,
        "acc",
        args.hidden_dim,
        args.patience,
    )

    print(f"{args.model.upper()} Accuracy: {acc:.4f}")
    np.save(args.output_path, acc)
