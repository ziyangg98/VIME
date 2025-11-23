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


def vime_semi(x_train, y_train, x_valid, y_valid, x_unlab, x_test, parameters: Dict[str, int],
              p_m: float, K: int, beta: float, file_name: str):
  """Semi-supervised learning part in VIME."""
  # Setup
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  hidden_dim = parameters['hidden_dim']
  batch_size = parameters['batch_size']
  iterations = parameters['iterations']

  # Load and apply encoder
  encoder = torch.load(file_name, weights_only=False).to(device).eval()

  # Pre-encode all data to avoid repeated CPU-GPU transfers
  with torch.no_grad():
    x_train_encoded = encoder(torch.from_numpy(x_train).float().to(device))
    x_valid_encoded = encoder(torch.from_numpy(x_valid).float().to(device))
    x_test_encoded = encoder(torch.from_numpy(x_test).float().to(device))

  # Build predictor
  predictor = Predictor(x_train_encoded.shape[1], hidden_dim, y_train.shape[1]).to(device)
  optimizer = torch.optim.Adam(predictor.parameters())

  # Setup early stopping
  os.makedirs('./save_model', exist_ok=True)
  model_path = './save_model/class_model.pth'
  best_loss = float('inf')
  patience_counter = 0
  patience = parameters.get('patience', 100)  # Match original VIME patience

  y_train_tensor = torch.from_numpy(y_train).float().to(device)
  y_valid_tensor = torch.from_numpy(y_valid).float().to(device)

  # Training loop
  for it in range(iterations):
    # Sample labeled batch (already encoded)
    batch_idx = np.random.permutation(len(x_train))[:batch_size]
    x_batch = x_train_encoded[batch_idx]
    y_batch = y_train_tensor[batch_idx]

    # Sample and augment unlabeled batch
    xu_batch_ori = x_unlab[np.random.permutation(len(x_unlab))[:batch_size]]
    xu_batch = []

    for _ in range(K):
      m_batch = mask_generator(p_m, xu_batch_ori)
      _, xu_temp = pretext_generator(m_batch, xu_batch_ori)
      with torch.no_grad():
        xu_temp = encoder(torch.from_numpy(xu_temp).float().to(device))
      xu_batch.append(xu_temp)

    xu_batch = torch.stack(xu_batch)  # [K, batch_size, dim]

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

    # Validation every iteration
    predictor.eval()
    with torch.no_grad():
      val_logit, _ = predictor(x_valid_encoded)
      val_loss = F.cross_entropy(val_logit, y_valid_tensor).item()

    # Print progress every 20 iterations
    if it % 20 == 0:
      print(f"  Iteration {it}/{iterations}, Sup Loss: {y_loss.item():.6f}, Unsup Loss: {yu_loss.item():.6f}, Val Loss: {val_loss:.6f}")

    # Early stopping (check every iteration)
    if val_loss < best_loss:
      best_loss = val_loss
      patience_counter = 0
      torch.save(predictor.state_dict(), model_path)
    elif (patience_counter := patience_counter + 1) >= patience:
      print(f"  Early stopping at iteration {it}")
      break

  # Load best model and predict
  predictor.load_state_dict(torch.load(model_path, weights_only=True))
  predictor.eval()

  with torch.no_grad():
    _, y_test_hat = predictor(x_test_encoded)
    return y_test_hat.cpu().numpy()
