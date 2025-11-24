"""Utility functions for VIME framework."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def generate_mask(p, x):
    """Generate binary mask matrix."""
    return np.random.binomial(1, p, x.shape)


def corrupt_samples(mask, x):
    """Generate corrupted samples by column-wise shuffling."""
    n, dim = x.shape
    x_shuffled = np.array(
        [x[np.random.permutation(n), i] for i in range(dim)]
    ).T
    x_corrupt = x * (1 - mask) + x_shuffled * mask
    return (x != x_corrupt).astype(int), x_corrupt


def evaluate(metric, y_true, y_pred):
    """Evaluate performance."""
    if metric == "acc":
        return accuracy_score(
            np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)
        )
    elif metric == "auc":
        return roc_auc_score(y_true[:, 1], y_pred[:, 1])


def to_labels(one_hot):
    """Convert one-hot matrix to label vector."""
    return np.argmax(one_hot, axis=1)


def to_one_hot(labels):
    """Convert label vector to one-hot matrix."""
    return np.eye(len(np.unique(labels)))[labels]


class EarlyStopping:
    """Early stopping helper."""

    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        """Check if should stop and update best model."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def load_best(self, model):
        """Load best model state."""
        if self.best_state:
            model.load_state_dict(self.best_state)


def log_progress(epoch, max_epochs, train_metrics, val_loss, interval=20):
    """Log training progress."""
    if (epoch + 1) % interval == 0:
        metrics_str = ", ".join(
            f"{k.capitalize()}: {v:.6f}" for k, v in train_metrics.items()
        )
        print(f"  Epoch {epoch+1}/{max_epochs}, {metrics_str}, Val: {val_loss:.6f}")


def log_iteration_progress(iteration, max_iterations, metrics, interval=20):
    """Log iteration progress."""
    if iteration % interval == 0:
        metrics_str = ", ".join(
            f"{k.capitalize()}: {v:.6f}" for k, v in metrics.items()
        )
        print(f"  Iter {iteration}/{max_iterations}, {metrics_str}")


def save_checkpoint(model, path):
    """Save model checkpoint."""
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, weights_only=True):
    """Load model checkpoint."""
    model.load_state_dict(torch.load(path, weights_only=weights_only))
