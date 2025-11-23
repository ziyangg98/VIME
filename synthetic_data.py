"""Synthetic data generation with latent variables.

Generate binary features x and binary labels y from latent variables z.
This creates a dataset where self-supervised learning should help.
"""

import numpy as np


def generate_synthetic_data(n_samples=10000, total_features=200, n_latent=10,
                           n_classes=2, noise_level=0.1, n_noise_features=0, seed=None):
  """Generate synthetic binary data from latent variables.

  Args:
    n_samples: number of samples to generate
    total_features: total dimensionality of x (fixed at 200)
    n_latent: dimensionality of latent variable z
    n_classes: number of classes for y
    noise_level: amount of noise in the generation process
    n_noise_features: number of irrelevant/noise features (rest are informative)
    seed: random seed for reproducibility

  Returns:
    x: binary feature matrix [n_samples, total_features]
    y: binary label matrix [n_samples, n_classes] (one-hot)
    z: latent variables [n_samples, n_latent]
  """
  if seed is not None:
    np.random.seed(seed)

  # Calculate number of informative features
  n_informative = total_features - n_noise_features

  # Generate latent variables z from standard normal
  z = np.random.randn(n_samples, n_latent)

  # Generate weight matrices
  # W_x maps latent z to informative feature logits
  W_x = np.random.randn(n_latent, n_informative) * 2
  # W_y maps latent z to class logits
  W_y = np.random.randn(n_latent, n_classes) * 2

  # Generate informative features x from latent z
  # x = z @ W_x + noise (continuous features)
  x_informative = z @ W_x + np.random.randn(n_samples, n_informative) * noise_level
  # Min-max normalization to [0, 1] range
  x_min = x_informative.min(axis=0, keepdims=True)
  x_max = x_informative.max(axis=0, keepdims=True)
  x_informative = (x_informative - x_min) / (x_max - x_min + 1e-8)

  # Generate labels y from latent z
  # y_logits = z @ W_y + noise
  y_logits = z @ W_y + np.random.randn(n_samples, n_classes) * noise_level
  # Apply softmax to get class probabilities
  y_probs = np.exp(y_logits - np.max(y_logits, axis=1, keepdims=True))
  y_probs = y_probs / np.sum(y_probs, axis=1, keepdims=True)
  # Sample class labels
  y_labels = np.array([np.random.choice(n_classes, p=p) for p in y_probs])
  # Convert to one-hot
  y = np.eye(n_classes)[y_labels]

  # Add noise features (irrelevant to y)
  if n_noise_features > 0:
    # Generate random noise features uniformly in [0, 1]
    x_noise = np.random.rand(n_samples, n_noise_features)
    # Concatenate informative and noise features
    x = np.concatenate([x_informative, x_noise], axis=1)
  else:
    x = x_informative

  return x, y, z


def load_synthetic_data(label_data_rate=0.1, n_samples=10000, total_features=200,
                       n_latent=10, n_classes=10, noise_level=0.1, n_noise_features=0, seed=42):
  """Load synthetic data with train/valid/test split.

  Args:
    label_data_rate: ratio of labeled data in training set
    n_samples: total number of samples
    total_features: total dimensionality of features (fixed at 200)
    n_latent: dimensionality of latent variables
    n_classes: number of classes
    noise_level: noise in generation process
    n_noise_features: number of irrelevant/noise features (rest are informative)
    seed: random seed

  Returns:
    x_label: labeled training features
    y_label: labeled training labels
    x_unlab: unlabeled training features
    x_valid: validation features
    y_valid: validation labels
    x_test: test features
    y_test: test labels
  """
  # Generate full dataset
  x_all, y_all, _ = generate_synthetic_data(
    n_samples=n_samples,
    total_features=total_features,
    n_latent=n_latent,
    n_classes=n_classes,
    noise_level=noise_level,
    n_noise_features=n_noise_features,
    seed=seed
  )

  # Split into train (64%), valid (16%), test (20%)
  n_train = int(n_samples * 0.64)
  n_valid = int(n_samples * 0.16)

  x_train = x_all[:n_train]
  y_train = y_all[:n_train]
  x_valid = x_all[n_train:n_train+n_valid]
  y_valid = y_all[n_train:n_train+n_valid]
  x_test = x_all[n_train+n_valid:]
  y_test = y_all[n_train+n_valid:]

  # Split training data into labeled and unlabeled
  # Set seed for reproducible splits
  if seed is not None:
    np.random.seed(seed + 1)  # Use different seed for splitting

  idx = np.random.permutation(len(y_train))
  label_idx = idx[:int(len(idx) * label_data_rate)]
  unlab_idx = idx[int(len(idx) * label_data_rate):]

  x_label = x_train[label_idx, :]
  y_label = y_train[label_idx, :]
  x_unlab = x_train[unlab_idx, :]

  return x_label, y_label, x_unlab, x_valid, y_valid, x_test, y_test


if __name__ == '__main__':
  # Test the data generation
  print("Testing synthetic data generation...")

  x_label, y_label, x_unlab, x_valid, y_valid, x_test, y_test = load_synthetic_data(
    label_data_rate=0.1,
    n_samples=10000,
    total_features=200,
    n_latent=10,
    n_classes=10,
    noise_level=0.1,
    n_noise_features=50,
    seed=42
  )

  print(f"\nDataset Statistics:")
  print(f"Labeled training samples: {x_label.shape[0]}")
  print(f"Unlabeled training samples: {x_unlab.shape[0]}")
  print(f"Validation samples: {x_valid.shape[0]}")
  print(f"Test samples: {x_test.shape[0]}")
  print(f"Total features: {x_label.shape[1]}")
  print(f"Informative features: {200 - 50}")
  print(f"Noise features: {50}")
  print(f"Number of classes: {y_label.shape[1]}")

  print(f"\nFeature statistics:")
  print(f"x_label mean: {x_label.mean():.3f}, std: {x_label.std():.3f}")
  print(f"x_unlab mean: {x_unlab.mean():.3f}, std: {x_unlab.std():.3f}")
  print(f"x_test mean: {x_test.mean():.3f}, std: {x_test.std():.3f}")

  print(f"\nLabel distribution (train):")
  for i in range(y_label.shape[1]):
    print(f"  Class {i}: {y_label[:, i].sum():.0f} samples ({y_label[:, i].mean()*100:.1f}%)")

  print(f"\nLabel distribution (test):")
  for i in range(y_test.shape[1]):
    print(f"  Class {i}: {y_test[:, i].sum():.0f} samples ({y_test[:, i].mean()*100:.1f}%)")
