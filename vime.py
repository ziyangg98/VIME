"""VIME self-supervised and semi-supervised learning."""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    MLP,
    EarlyStopping,
    corrupt_samples,
    generate_mask,
    log_iteration_progress,
    log_progress,
)


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
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-7)

    # Generate corrupted samples
    mask_train, mask_val = generate_mask(p_mask, X_train), generate_mask(p_mask, X_val)
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
    encoder: nn.Module,
):
    """VIME semi-supervised learning."""
    device = next(encoder.parameters()).device

    # Pre-encode data
    with torch.no_grad():
        X_train_enc = encoder(torch.from_numpy(X_train).float().to(device))
        X_val_enc = encoder(torch.from_numpy(X_val).float().to(device))
        X_test_enc = encoder(torch.from_numpy(X_test).float().to(device))

    predictor = MLP(X_train_enc.shape[1], y_train.shape[1], params["hidden_dim"]).to(
        device
    )
    optimizer = torch.optim.Adam(predictor.parameters())
    early_stopping = EarlyStopping(patience=params.get("patience", 5))

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
        y_logit = predictor(X_batch)
        sup_loss = F.cross_entropy(y_logit, y_batch)

        k, bs, dim = X_u_batch.shape
        y_u_logit = predictor(X_u_batch.view(-1, dim)).view(k, bs, -1)
        unsup_loss = torch.var(y_u_logit, dim=0).mean()

        loss = sup_loss + beta * unsup_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        predictor.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(predictor(X_val_enc), y_val_t).item()

        log_iteration_progress(
            i,
            params["iterations"],
            {"sup": sup_loss.item(), "unsup": unsup_loss.item(), "val": val_loss},
        )

        if early_stopping.step(val_loss, predictor):
            print(f"  Early stopping at iteration {i}")
            break

    early_stopping.load_best(predictor)
    predictor.eval()
    with torch.no_grad():
        return F.softmax(predictor(X_test_enc), dim=1).cpu().numpy()


if __name__ == "__main__":
    import argparse

    from utils import evaluate, load_data, set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data_cache/data.npz")
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--semi_patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--p_m", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semi", action="store_true", help="Run VIME-Semi (default: VIME-Self + MLP)")
    parser.add_argument("--output_path", type=str, default="./results/vime.npy")
    args = parser.parse_args()

    from baseline import train_supervised
    from utils import transform_with_encoder

    set_seed(args.seed)
    data = load_data(args.data_path)

    # Train VIME-Self encoder
    print("Training VIME-Self encoder...")
    X_all = np.vstack([data["X_train"], data["X_unlabeled"]])
    encoder = vime_self(
        X_all,
        data["X_val"],
        args.p_m,
        args.alpha,
        {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "hidden_dim": data["X_train"].shape[1],
            "patience": args.patience,
        },
    )

    if args.semi:
        # Train VIME-Semi
        print("Training VIME-Semi...")
        y_pred = vime_semi(
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            data["X_unlabeled"],
            data["X_test"],
            {
                "hidden_dim": args.hidden_dim,
                "batch_size": args.batch_size,
                "iterations": args.iterations,
                "patience": args.semi_patience,
            },
            args.p_m,
            args.K,
            args.beta,
            encoder,
        )
        acc = evaluate("acc", data["y_test"], y_pred)
        print(f"VIME-Semi Accuracy: {acc:.4f}")
    else:
        # VIME-Self + MLP
        print("Evaluating VIME-Self + MLP...")
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
        print(f"VIME-Self + MLP Accuracy: {acc:.4f}")

    np.save(args.output_path, acc)
