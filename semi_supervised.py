"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar,
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain,"
Neural Information Processing Systems (NeurIPS), 2020.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from utils import mask_generator, pretext_generator


class Predictor(nn.Module):
  """Predictor network for semi-supervised learning."""

  def __init__(self, input_dim: int, hidden_dim: int, label_dim: int):
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(input_dim, hidden_dim), nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
      nn.Linear(hidden_dim, label_dim)
    )

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logit = self.network(x)
    return logit, F.softmax(logit, dim=-1)


def vime_semi(x_train, y_train, x_unlab, x_test, parameters: Dict[str, int],
              p_m: float, K: int, beta: float, file_name: str):
  """Semi-supervised learning part in VIME."""
  # Setup
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  hidden_dim = parameters['hidden_dim']
  batch_size = parameters['batch_size']
  iterations = parameters['iterations']

  # Split train/validation
  idx = np.random.permutation(len(x_train))
  split = int(len(idx) * 0.9)
  x_train, x_valid = x_train[idx[:split]], x_train[idx[split:]]
  y_train, y_valid = y_train[idx[:split]], y_train[idx[split:]]

  # Load and apply encoder
  encoder = torch.load(file_name, weights_only=False).to(device).eval()

  with torch.no_grad():
    x_valid = encoder(torch.from_numpy(x_valid).float().to(device)).cpu().numpy()
    x_test = encoder(torch.from_numpy(x_test).float().to(device)).cpu().numpy()

  # Build predictor
  predictor = Predictor(x_valid.shape[1], hidden_dim, y_train.shape[1]).to(device)
  optimizer = torch.optim.Adam(predictor.parameters())

  # Setup early stopping
  os.makedirs('./save_model', exist_ok=True)
  model_path = './save_model/class_model.pth'
  best_loss = float('inf')
  best_idx = -1

  y_valid_tensor = torch.from_numpy(y_valid).float().to(device)

  # Training loop
  for it in range(iterations):
    # Sample labeled batch
    batch_idx = np.random.permutation(len(x_train))[:batch_size]
    x_batch, y_batch = x_train[batch_idx], y_train[batch_idx]

    # Encode labeled batch
    with torch.no_grad():
      x_batch = encoder(torch.from_numpy(x_batch).float().to(device)).cpu().numpy()

    # Sample and augment unlabeled batch
    xu_batch_ori = x_unlab[np.random.permutation(len(x_unlab))[:batch_size]]
    xu_batch = []

    for _ in range(K):
      m_batch = mask_generator(p_m, xu_batch_ori)
      _, xu_temp = pretext_generator(m_batch, xu_batch_ori)
      with torch.no_grad():
        xu_temp = encoder(torch.from_numpy(xu_temp).float().to(device)).cpu().numpy()
      xu_batch.append(xu_temp)

    xu_batch = np.array(xu_batch)  # [K, batch_size, dim]

    # Convert to tensors and move to device
    x_batch = torch.from_numpy(x_batch).float().to(device)
    y_batch = torch.from_numpy(y_batch).float().to(device)
    xu_batch = torch.from_numpy(xu_batch).float().to(device)

    # Train predictor
    predictor.train()

    # Supervised loss
    y_logit, _ = predictor(x_batch)
    y_loss = F.cross_entropy(y_logit, y_batch)

    # Unsupervised loss (variance across augmentations)
    K_size, bs, dim = xu_batch.shape
    yv_logit, _ = predictor(xu_batch.view(-1, dim))
    yv_logit = yv_logit.view(K_size, bs, -1)
    yu_loss = torch.var(yv_logit, dim=0).mean()

    # Combined loss
    loss = y_loss + beta * yu_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Validation
    if it % 100 == 0:
      predictor.eval()
      with torch.no_grad():
        val_logit, _ = predictor(torch.from_numpy(x_valid).float().to(device))
        val_loss = F.cross_entropy(val_logit, y_valid_tensor).item()

      # Early stopping
      if val_loss < best_loss:
        best_loss = val_loss
        best_idx = it
        torch.save(predictor.state_dict(), model_path)
      elif best_idx + 100 < it:
        break

  # Load best model and predict
  predictor.load_state_dict(torch.load(model_path, weights_only=True))
  predictor.eval()

  with torch.no_grad():
    _, y_test_hat = predictor(torch.from_numpy(x_test).float().to(device))
    return y_test_hat.cpu().numpy()
