"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar,
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain,"
Neural Information Processing Systems (NeurIPS), 2020.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from utils import mask_generator, pretext_generator


class VIMESelfModel(nn.Module):
  """VIME Self-supervised model."""

  def __init__(self, dim: int):
    super().__init__()
    self.encoder = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
    self.mask_estimator = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
    self.feature_estimator = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = self.encoder(x)
    return self.mask_estimator(encoded), self.feature_estimator(encoded)


def vime_self(x_unlab, p_m: float, alpha: float, parameters: Dict[str, int]) -> nn.Module:
  """Self-supervised learning part in VIME.

  Args:
    x_unlab: unlabeled features (numpy array)
    p_m: corruption probability
    alpha: weight for feature loss vs mask loss
    parameters: dict with 'epochs' and 'batch_size'

  Returns:
    encoder: trained encoder module
  """
  # Setup
  _, dim = x_unlab.shape
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Build model
  model = VIMESelfModel(dim).to(device)
  # Use Keras-compatible RMSprop parameters
  optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-7)

  # Generate corrupted samples (on CPU)
  m_unlab = mask_generator(p_m, x_unlab)
  m_label, x_tilde = pretext_generator(m_unlab, x_unlab)

  # Create dataset (keep on CPU, move to GPU in training loop)
  dataset = TensorDataset(
    torch.from_numpy(x_tilde).float(),
    torch.from_numpy(m_label).float(),
    torch.from_numpy(x_unlab).float()
  )
  dataloader = DataLoader(
    dataset,
    batch_size=parameters['batch_size'],
    shuffle=True,
    pin_memory=True if device.type == 'cuda' else False
  )

  # Training loop
  model.train()
  for _ in range(parameters['epochs']):
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

  # Return encoder in eval mode
  return model.encoder.eval()
