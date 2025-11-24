"""VIME semi-supervised learning."""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from utils import (
    generate_mask,
    corrupt_samples,
    EarlyStopping,
    log_iteration_progress,
    save_checkpoint,
    load_checkpoint,
)


class Predictor(nn.Module):
    """Predictor network."""

    def __init__(self, input_dim: int, hidden_dim: int, label_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, label_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logit = self.network(x)
        return logit, F.softmax(logit, dim=-1)


def vime_semi(
    X_train,
    y_train,
    X_val,
    y_val,
    X_unlabeled,
    X_test,
    params: Dict[str, int],
    p_mask: float,
    K: int,
    beta: float,
    encoder_path: str,
):
    """VIME semi-supervised learning."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.load(encoder_path, weights_only=False).to(device).eval()

    # Pre-encode data
    with torch.no_grad():
        X_train_enc = encoder(torch.from_numpy(X_train).float().to(device))
        X_val_enc = encoder(torch.from_numpy(X_val).float().to(device))
        X_test_enc = encoder(torch.from_numpy(X_test).float().to(device))

    predictor = Predictor(
        X_train_enc.shape[1], params["hidden_dim"], y_train.shape[1]
    ).to(device)
    optimizer = torch.optim.Adam(predictor.parameters())

    os.makedirs("./save_model", exist_ok=True)
    model_path = "./save_model/class_model.pth"
    early_stopping = EarlyStopping(patience=params.get("patience", 100))

    y_train_t = torch.from_numpy(y_train).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    for i in range(params["iterations"]):
        # Sample labeled batch
        idx = np.random.permutation(len(X_train))[: params["batch_size"]]
        X_batch, y_batch = X_train_enc[idx], y_train_t[idx]

        # Sample and augment unlabeled batch
        X_u = X_unlabeled[
            np.random.permutation(len(X_unlabeled))[: params["batch_size"]]
        ]
        X_u_batch = torch.stack(
            [
                encoder(
                    torch.from_numpy(
                        corrupt_samples(generate_mask(p_mask, X_u), X_u)[1]
                    )
                    .float()
                    .to(device)
                )
                for _ in range(K)
            ]
        )

        predictor.train()
        y_logit, _ = predictor(X_batch)
        sup_loss = F.cross_entropy(y_logit, y_batch)

        k, bs, dim = X_u_batch.shape
        y_u_logit = predictor(X_u_batch.view(-1, dim))[0].view(k, bs, -1)
        unsup_loss = torch.var(y_u_logit, dim=0).mean()

        loss = sup_loss + beta * unsup_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        predictor.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(predictor(X_val_enc)[0], y_val_t).item()

        log_iteration_progress(
            i,
            params["iterations"],
            {"sup": sup_loss.item(), "unsup": unsup_loss.item(), "val": val_loss},
        )

        if early_stopping.step(val_loss, predictor):
            save_checkpoint(predictor, model_path)
            print(f"  Early stopping at iteration {i}")
            break

        if val_loss == early_stopping.best_loss:
            save_checkpoint(predictor, model_path)

    load_checkpoint(predictor, model_path)
    predictor.eval()
    with torch.no_grad():
        return predictor(X_test_enc)[1].cpu().numpy()
