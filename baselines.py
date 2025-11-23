"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar,
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain,"
Neural Information Processing Systems (NeurIPS), 2020.
"""

from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from utils import convert_matrix_to_vector, convert_vector_to_matrix


def logit(x_train, y_train, x_test):
  """Logistic Regression."""
  if len(y_train.shape) > 1:
    y_train = convert_matrix_to_vector(y_train)
  model = LogisticRegression(max_iter=1000)
  model.fit(x_train, y_train)
  return model.predict_proba(x_test)


def xgb_model(x_train, y_train, x_test):
  """XGBoost."""
  if len(y_train.shape) > 1:
    y_train = convert_matrix_to_vector(y_train)
  model = xgb.XGBClassifier()
  model.fit(x_train, y_train)
  return model.predict_proba(x_test)


class MLPModel(nn.Module):
  """Multi-layer perceptron model."""

  def __init__(self, data_dim: int, label_dim: int, hidden_dim: int, activation: str = 'relu'):
    super().__init__()
    act_fn = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}.get(activation, nn.ReLU())
    self.network = nn.Sequential(
      nn.Linear(data_dim, hidden_dim), act_fn,
      nn.Linear(hidden_dim, hidden_dim), act_fn,
      nn.Linear(hidden_dim, label_dim)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.network(x)


def mlp(x_train, y_train, x_valid, y_valid, x_test, parameters: Dict[str, int]):
  """Multi-layer perceptron (MLP)."""
  if len(y_train.shape) == 1:
    y_train = convert_vector_to_matrix(y_train)
  if len(y_valid.shape) == 1:
    y_valid = convert_vector_to_matrix(y_valid)

  # Setup
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = MLPModel(
    x_train.shape[1],
    y_train.shape[1],
    parameters['hidden_dim'],
    parameters['activation']
  ).to(device)
  optimizer = torch.optim.Adam(model.parameters())

  # Create dataloader (keep data on CPU)
  train_loader = DataLoader(
    TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()),
    batch_size=parameters['batch_size'],
    shuffle=True,
    pin_memory=device.type == 'cuda'
  )

  y_valid = torch.from_numpy(y_valid).float().to(device)

  # Training with early stopping
  best_loss = float('inf')
  patience_counter = 0
  patience = parameters.get('patience', 5)
  best_state = None

  for epoch in range(parameters['epochs']):
    # Training
    model.train()
    epoch_train_loss = 0.0
    num_batches = 0
    for batch_x, batch_y in train_loader:
      batch_x = batch_x.to(device, non_blocking=True)
      batch_y = batch_y.to(device, non_blocking=True)

      logits = model(batch_x)
      loss = F.cross_entropy(logits, batch_y)

      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

      epoch_train_loss += loss.item()
      num_batches += 1

    # Validation
    model.eval()
    with torch.no_grad():
      val_logits = model(torch.from_numpy(x_valid).float().to(device))
      val_loss = F.cross_entropy(val_logits, y_valid).item()

    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
      avg_train_loss = epoch_train_loss / num_batches
      print(f"  Epoch {epoch+1}/{parameters['epochs']}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Early stopping
    if val_loss < best_loss:
      best_loss = val_loss
      patience_counter = 0
      best_state = model.state_dict()
    elif (patience_counter := patience_counter + 1) >= patience:
      print(f"  Early stopping at epoch {epoch+1}")
      break

  # Restore best model and predict
  if best_state:
    model.load_state_dict(best_state)

  model.eval()
  with torch.no_grad():
    logits = model(torch.from_numpy(x_test).float().to(device))
    return F.softmax(logits, dim=1).cpu().numpy()
