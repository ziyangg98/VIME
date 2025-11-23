"""Autoencoder-based Self-supervised Learning.

Simplified version of VIME-Self without mask prediction.
Same architecture but only uses reconstruction loss.
"""

from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class AutoencoderModel(nn.Module):
  """Autoencoder model - simplified VIME-Self without mask prediction."""

  def __init__(self, dim: int):
    super().__init__()
    # Encoder (same as VIME-Self)
    self.encoder = nn.Sequential(
      nn.Linear(dim, dim),
      nn.ReLU()
    )
    # Feature estimator/decoder (same as VIME-Self)
    self.feature_estimator = nn.Sequential(
      nn.Linear(dim, dim),
      nn.Sigmoid()
    )

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = self.encoder(x)
    feature_output = self.feature_estimator(encoded)
    return feature_output, encoded


def train_autoencoder(x_unlab, parameters: Dict[str, int]) -> nn.Module:
  """Train autoencoder - simple reconstruction without mask or corruption.

  Args:
    x_unlab: unlabeled feature
    parameters: epochs, batch_size

  Returns:
    encoder: Representation learning block
  """
  # Parameters (same as vime_self)
  _, dim = x_unlab.shape
  epochs = parameters['epochs']
  batch_size = parameters['batch_size']
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Build model (same structure as VIME-Self)
  model = AutoencoderModel(dim).to(device)
  # Use same optimizer settings as VIME-Self
  optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-7)

  # Define loss functions
  mse_loss = nn.MSELoss()

  # Convert to tensors (same as vime_self)
  x_unlab_tensor = torch.from_numpy(x_unlab).float()

  # Create dataset (same as vime_self)
  dataset = TensorDataset(x_unlab_tensor)
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True if device.type == 'cuda' else False
  )

  # Training loop (same as vime_self)
  model.train()
  for _ in range(epochs):
    for (batch_x,) in dataloader:
      # Move batch to device
      batch_x = batch_x.to(device, non_blocking=True)

      # Forward pass
      feature_pred, _ = model(batch_x)

      # Compute loss (only reconstruction, no mask)
      loss = mse_loss(feature_pred, batch_x)

      # Backward pass
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

  # Return encoder in eval mode (same as vime_self)
  return model.encoder.eval()
