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

  def __init__(self, input_dim: int, hidden_dim: int):
    super().__init__()
    # Encoder
    self.encoder = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.ReLU()
    )
    # Decoder
    self.decoder = nn.Sequential(
      nn.Linear(hidden_dim, input_dim),
      nn.Sigmoid()
    )

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = self.encoder(x)
    reconstructed = self.decoder(encoded)
    return reconstructed, encoded


def train_autoencoder(x_train, x_valid, parameters: Dict[str, int]) -> nn.Module:
  """Train autoencoder - simple reconstruction without mask or corruption.

  Args:
    x_train: training features (labeled + unlabeled combined)
    x_valid: validation features (external, for early stopping)
    parameters: epochs, batch_size, hidden_dim

  Returns:
    encoder: Representation learning block
  """
  # Parameters
  _, dim = x_train.shape
  epochs = parameters['epochs']
  batch_size = parameters['batch_size']
  hidden_dim = parameters['hidden_dim']
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Build model
  model = AutoencoderModel(dim, hidden_dim).to(device)
  # Use same optimizer settings as VIME-Self
  optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-7)

  # Define loss function
  mse_loss = nn.MSELoss()

  # Convert to tensors (same as vime_self)
  x_train_tensor = torch.from_numpy(x_train).float()
  x_valid_tensor = torch.from_numpy(x_valid).float().to(device)

  # Create dataset (same as vime_self)
  dataset = TensorDataset(x_train_tensor)
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True if device.type == 'cuda' else False
  )

  # Training loop with early stopping
  best_loss = float('inf')
  patience_counter = 0
  patience = parameters.get('patience', 5)
  best_state = None

  for epoch in range(epochs):
    # Training
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    for (batch_x,) in dataloader:
      # Move batch to device
      batch_x = batch_x.to(device, non_blocking=True)

      # Forward pass
      reconstructed, _ = model(batch_x)

      # Compute loss (only reconstruction, no mask)
      loss = mse_loss(reconstructed, batch_x)

      # Backward pass
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
      num_batches += 1

    # Validation every epoch
    model.eval()
    with torch.no_grad():
      valid_reconstructed, _ = model(x_valid_tensor)
      val_loss = mse_loss(valid_reconstructed, x_valid_tensor).item()

    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
      avg_train_loss = epoch_loss / num_batches
      print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

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

  # Return encoder in eval mode (same as vime_self)
  return model.encoder.eval()
