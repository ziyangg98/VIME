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

from utils import mask_generator, pretext_generator


class VIMESelfModel(nn.Module):
  """VIME Self-supervised model."""

  def __init__(self, input_dim: int, hidden_dim: int):
    super().__init__()
    self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
    self.mask_estimator = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())
    self.feature_estimator = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = self.encoder(x)
    return self.mask_estimator(encoded), self.feature_estimator(encoded)


def vime_self(x_train, x_valid, p_m: float, alpha: float, parameters: Dict[str, int]) -> nn.Module:
  """Self-supervised learning part in VIME.

  Args:
    x_train: training features (labeled + unlabeled combined)
    x_valid: validation features (external, for early stopping)
    p_m: corruption probability
    alpha: weight for feature loss vs mask loss
    parameters: dict with 'epochs', 'batch_size', 'hidden_dim'

  Returns:
    encoder: trained encoder module
  """
  # Setup
  _, dim = x_train.shape
  hidden_dim = parameters['hidden_dim']
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Build model
  model = VIMESelfModel(dim, hidden_dim).to(device)
  # Use Keras-compatible RMSprop parameters
  optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-7)

  # Generate corrupted samples for training (on CPU)
  m_train = mask_generator(p_m, x_train)
  m_label, x_tilde = pretext_generator(m_train, x_train)

  # Generate corrupted samples for validation
  m_valid = mask_generator(p_m, x_valid)
  m_label_valid, x_tilde_valid = pretext_generator(m_valid, x_valid)

  # Create training dataset (keep on CPU, move to GPU in training loop)
  dataset = TensorDataset(
    torch.from_numpy(x_tilde).float(),
    torch.from_numpy(m_label).float(),
    torch.from_numpy(x_train).float()
  )
  dataloader = DataLoader(
    dataset,
    batch_size=parameters['batch_size'],
    shuffle=True,
    pin_memory=True if device.type == 'cuda' else False
  )

  # Prepare validation data
  x_tilde_valid_tensor = torch.from_numpy(x_tilde_valid).float().to(device)
  m_label_valid_tensor = torch.from_numpy(m_label_valid).float().to(device)
  x_valid_tensor = torch.from_numpy(x_valid).float().to(device)

  # Training loop with early stopping
  best_loss = float('inf')
  patience_counter = 0
  patience = parameters.get('patience', 5)
  best_state = None

  for epoch in range(parameters['epochs']):
    # Training
    model.train()
    epoch_loss = 0.0
    epoch_mask_loss = 0.0
    epoch_feature_loss = 0.0
    num_batches = 0
    for batch_x_tilde, batch_m_label, batch_x_unlab in dataloader:
      # Move batch to device
      batch_x_tilde = batch_x_tilde.to(device, non_blocking=True)
      batch_m_label = batch_m_label.to(device, non_blocking=True)
      batch_x_unlab = batch_x_unlab.to(device, non_blocking=True)

      # Forward pass
      mask_pred, feature_pred = model(batch_x_tilde)

      # Compute loss using functional API
      mask_loss = F.binary_cross_entropy(mask_pred, batch_m_label)
      feature_loss = F.mse_loss(feature_pred, batch_x_unlab)
      loss = mask_loss + alpha * feature_loss

      # Backward pass
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
      epoch_mask_loss += mask_loss.item()
      epoch_feature_loss += feature_loss.item()
      num_batches += 1

    # Validation every epoch
    model.eval()
    with torch.no_grad():
      mask_pred_val, feature_pred_val = model(x_tilde_valid_tensor)
      val_mask_loss = F.binary_cross_entropy(mask_pred_val, m_label_valid_tensor).item()
      val_feature_loss = F.mse_loss(feature_pred_val, x_valid_tensor).item()
      val_loss = val_mask_loss + alpha * val_feature_loss

    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
      avg_train_loss = epoch_loss / num_batches
      avg_mask = epoch_mask_loss / num_batches
      avg_feature = epoch_feature_loss / num_batches
      print(f"  Epoch {epoch+1}/{parameters['epochs']}, Train Loss: {avg_train_loss:.6f} (Mask: {avg_mask:.6f}, Feature: {avg_feature:.6f}), Val Loss: {val_loss:.6f}")

    # Early stopping (check every epoch)
    if val_loss < best_loss:
      best_loss = val_loss
      patience_counter = 0
      best_state = model.state_dict()
    elif (patience_counter := patience_counter + 1) >= patience:
      print(f"  Early stopping at epoch {epoch+1}")
      break

  # Restore best model
  if best_state is not None:
    model.load_state_dict(best_state)

  # Return encoder in eval mode
  return model.encoder.eval()
