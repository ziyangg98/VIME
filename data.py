"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

data.py
- Load and preprocess MNIST data (http://yann.lecun.com/exdb/mnist/)
"""

# Necessary packages
import numpy as np
import pandas as pd
from torchvision import datasets

def load_mnist_data(label_data_rate):
  """MNIST data loading with train/valid/test split.

  Args:
    - label_data_rate: ratio of labeled data in training set

  Returns:
    - x_label, y_label: labeled training dataset
    - x_unlab: unlabeled training dataset
    - x_valid, y_valid: validation dataset
    - x_test, y_test: test dataset
  """
  # Import mnist data
  train_dataset = datasets.MNIST(root='./data', train=True, download=True)
  test_dataset = datasets.MNIST(root='./data', train=False, download=True)

  x_train_all = train_dataset.data.numpy()
  y_train_all = train_dataset.targets.numpy()
  x_test = test_dataset.data.numpy()
  y_test = test_dataset.targets.numpy()

  # One hot encoding for the labels
  y_train_all = np.asarray(pd.get_dummies(y_train_all))
  y_test = np.asarray(pd.get_dummies(y_test))

  # Normalize features
  x_train_all = x_train_all / 255.0
  x_test = x_test / 255.0

  # Treat MNIST data as tabular data with 784 features
  no, dim_x, dim_y = np.shape(x_train_all)
  test_no, _, _ = np.shape(x_test)

  x_train_all = np.reshape(x_train_all, [no, dim_x * dim_y])
  x_test = np.reshape(x_test, [test_no, dim_x * dim_y])

  # Split train_all into train (80%) and valid (20%)
  n_train = int(no * 0.8)
  x_train = x_train_all[:n_train]
  y_train = y_train_all[:n_train]
  x_valid = x_train_all[n_train:]
  y_valid = y_train_all[n_train:]

  # Divide training data into labeled and unlabeled
  idx = np.random.permutation(len(y_train))

  # Label data : Unlabeled data = label_data_rate:(1-label_data_rate)
  label_idx = idx[:int(len(idx)*label_data_rate)]
  unlab_idx = idx[int(len(idx)*label_data_rate):]

  # Unlabeled data
  x_unlab = x_train[unlab_idx, :]

  # Labeled data
  x_label = x_train[label_idx, :]
  y_label = y_train[label_idx, :]

  return x_label, y_label, x_unlab, x_valid, y_valid, x_test, y_test