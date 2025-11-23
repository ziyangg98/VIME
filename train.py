"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

train.py
- Main training script for VIME framework
- Trains and evaluates supervised, self-supervised, and semi-supervised models
"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
import random

from data import load_mnist_data
from synthetic_data import load_synthetic_data
from baselines import logit, xgb_model, mlp
from self_supervised import vime_self
from semi_supervised import vime_semi
from autoencoder import train_autoencoder
from utils import perf_metric


def set_seed(seed):
  """Set random seeds for reproducibility."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  # Make PyTorch deterministic
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


#%%
def supervised_model_training(x_train, y_train, x_valid, y_valid, x_test,
                              y_test, model_name, metric, hidden_dim=100, patience=50):
  """Train supervised learning models and report the results.

  Args:
    - x_train, y_train: training dataset
    - x_valid, y_valid: validation dataset
    - x_test, y_test: testing dataset
    - model_name: logit, xgboost, or mlp
    - metric: acc or auc
    - hidden_dim: hidden layer dimension for MLP
    - patience: early stopping patience for MLP

  Returns:
    - performance: prediction performance
  """

  # Train supervised model
  # Logistic regression
  if model_name == 'logit':
    # Combine train+valid for logit and xgboost (they don't use validation)
    x_train_full = np.vstack([x_train, x_valid])
    y_train_full = np.vstack([y_train, y_valid])
    y_test_hat = logit(x_train_full, y_train_full, x_test)
  # XGBoost
  elif model_name == 'xgboost':
    x_train_full = np.vstack([x_train, x_valid])
    y_train_full = np.vstack([y_train, y_valid])
    y_test_hat = xgb_model(x_train_full, y_train_full, x_test)
  # MLP
  elif model_name == 'mlp':
    mlp_parameters = dict()
    mlp_parameters['hidden_dim'] = hidden_dim
    mlp_parameters['epochs'] = 1000
    mlp_parameters['activation'] = 'relu'
    mlp_parameters['batch_size'] = 100
    mlp_parameters['patience'] = patience

    y_test_hat = mlp(x_train, y_train, x_valid, y_valid, x_test, mlp_parameters)
  else:
    raise ValueError(f"Unknown model name: {model_name}. Choose 'logit', 'xgboost', or 'mlp'.")

  # Report the performance
  performance = perf_metric(metric, y_test, y_test_hat)    
    
  return performance    

#%%
def vime_main (label_data_rate, model_sets, label_no, p_m, alpha, K, beta, dataset='mnist', n_noise_features=0, hidden_dim=100, patience=50):
  """VIME Main function.

  Args:
    - model_sets: supervised model sets
    - label_no: number of labeled data to be used
    - p_m: corruption probability
    - alpha: hyper-parameter to control two self-supervied loss
    - K: number of augmented data
    - beta: hyper-parameter to control two semi-supervied loss
    - dataset: 'mnist' or 'synthetic'

  Returns:
    - results: performances of supervised, autoencoder, VIME-self and VIME-semi performance
  """

  # Define outputs (supervised, autoencoder, VIME-Self, VIME-Semi)
  results = np.zeros([len(model_sets)+3])

  # Load data
  if dataset == 'mnist':
    x_train, y_train, x_unlab, x_valid, y_valid, x_test, y_test = load_mnist_data(label_data_rate)
  elif dataset == 'synthetic':
    x_train, y_train, x_unlab, x_valid, y_valid, x_test, y_test = load_synthetic_data(
      label_data_rate=label_data_rate,
      total_features=200,
      n_noise_features=n_noise_features
    )
  else:
    raise ValueError(f"Unknown dataset: {dataset}. Choose 'mnist' or 'synthetic'.")

  # Use subset of labeled data
  x_train = x_train[:label_no, :]
  y_train = y_train[:label_no, :]

  # Get input dimension for encoder
  input_dim = x_train.shape[1]

  # Metric
  metric = 'acc'

  # Train supervised models
  print("\n" + "="*70)
  print("STEP 1: Training Supervised Baseline ({})".format(model_sets[0].upper()))
  print("="*70)
  for m_it in range(len(model_sets)):
    model_name = model_sets[m_it]
    results[m_it] = supervised_model_training(x_train, y_train,
                                              x_valid, y_valid,
                                              x_test, y_test, model_name, metric,
                                              hidden_dim, patience)
  print("Supervised baseline accuracy: {:.4f}".format(results[0]))

  # Train Autoencoder
  print("\n" + "="*70)
  print("STEP 2: Training Autoencoder")
  print("="*70)
  # Combine labeled and unlabeled data for self-supervised pretraining
  x_train_all = np.vstack([x_train, x_unlab])
  autoencoder_parameters = dict()
  autoencoder_parameters['batch_size'] = 128
  autoencoder_parameters['epochs'] = 1000
  autoencoder_parameters['hidden_dim'] = input_dim  # Use input_dim for encoder
  autoencoder_parameters['patience'] = patience
  # Self-supervised learning on all training data (labeled + unlabeled)
  autoencoder_encoder = train_autoencoder(x_train_all, x_valid, autoencoder_parameters)

  # Save autoencoder encoder
  autoencoder_file_name = './save_model/autoencoder_encoder.pth'
  torch.save(autoencoder_encoder, autoencoder_file_name)

  # Test Autoencoder
  print("\n" + "-"*70)
  print("Testing Autoencoder + MLP classifier")
  print("-"*70)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  with torch.no_grad():
    x_train_ae = autoencoder_encoder(torch.from_numpy(x_train).float().to(device)).cpu().numpy()
    x_valid_ae = autoencoder_encoder(torch.from_numpy(x_valid).float().to(device)).cpu().numpy()
    x_test_ae = autoencoder_encoder(torch.from_numpy(x_test).float().to(device)).cpu().numpy()

  model_name = 'mlp'
  results[len(model_sets)] = supervised_model_training(x_train_ae, y_train,
                                                        x_valid_ae, y_valid,
                                                        x_test_ae, y_test,
                                                        model_name,
                                                        metric,
                                                        hidden_dim, patience)
  print("Autoencoder + MLP accuracy: {:.4f}".format(results[len(model_sets)]))

  # Train VIME-Self
  print("\n" + "="*70)
  print("STEP 3: Training VIME-Self")
  print("="*70)
  vime_self_parameters = dict()
  vime_self_parameters['batch_size'] = 128
  vime_self_parameters['epochs'] = 1000
  vime_self_parameters['hidden_dim'] = input_dim  # Use input_dim for encoder
  vime_self_parameters['patience'] = patience
  # Self-supervised learning on all training data (labeled + unlabeled)
  vime_self_encoder = vime_self(x_train_all, x_valid, p_m, alpha, vime_self_parameters)

  # Save encoder
  os.makedirs('save_model', exist_ok=True)
  file_name = './save_model/encoder_model.pth'
  torch.save(vime_self_encoder, file_name)

  # Test VIME-Self
  print("\n" + "-"*70)
  print("Testing VIME-Self + MLP classifier")
  print("-"*70)

  with torch.no_grad():
    x_train_hat = vime_self_encoder(torch.from_numpy(x_train).float().to(device)).cpu().numpy()
    x_valid_hat = vime_self_encoder(torch.from_numpy(x_valid).float().to(device)).cpu().numpy()
    x_test_hat = vime_self_encoder(torch.from_numpy(x_test).float().to(device)).cpu().numpy()

  model_name = 'mlp'
  results[len(model_sets)+1] = supervised_model_training(x_train_hat, y_train,
                                                          x_valid_hat, y_valid,
                                                          x_test_hat, y_test,
                                                          model_name,
                                                          metric,
                                                          hidden_dim, patience)
  print("VIME-Self + MLP accuracy: {:.4f}".format(results[len(model_sets)+1]))

  # Train VIME-Semi
  print("\n" + "="*70)
  print("STEP 4: Training VIME-Semi (End-to-End)")
  print("="*70)
  vime_semi_parameters = dict()
  vime_semi_parameters['hidden_dim'] = hidden_dim
  vime_semi_parameters['batch_size'] = 128
  vime_semi_parameters['iterations'] = 1000
  vime_semi_parameters['patience'] = 100
  y_test_hat = vime_semi(x_train, y_train, x_valid, y_valid,
                         x_unlab, x_test, vime_semi_parameters, p_m, K, beta, file_name)

  # Test VIME-Semi
  results[len(model_sets)+2] = perf_metric(metric, y_test, y_test_hat)
  print("VIME-Semi accuracy: {:.4f}".format(results[len(model_sets)+2]))

  return results


def exp_main(args):
  """Main function for experiments.

  Args:
    - iterations: Number of experiments iterations
    - label_no: Number of labeled data to be used
    - model_name: supervised model name (mlp, logit, or xgboost)
    - p_m: corruption probability for self-supervised learning
    - alpha: hyper-parameter to control the weights of feature and mask losses
    - K: number of augmented samples
    - beta: hyperparameter to control supervised and unsupervised loss
    - label_data_rate: ratio of labeled data
    - seed: random seed for reproducibility

  Returns:
    - results: performances of 4 different models (supervised only, autoencoder, VIME-self, and VIME)
  """

  # Define output
  results = np.zeros([args.iterations, 4])

  # Iterations - each with different seed for reproducibility
  for it in range(args.iterations):
    # Set different seed for each iteration
    set_seed(args.seed + it)

    print("\n" + "#"*70)
    print("# ITERATION {}/{}".format(it+1, args.iterations))
    print("#"*70)

    results[it, :] = vime_main(args.label_data_rate,
                               [args.model_name],
                               args.label_no,
                               args.p_m,
                               args.alpha,
                               args.K,
                               args.beta,
                               args.dataset,
                               args.n_noise_features,
                               args.hidden_dim,
                               args.patience)

    print("\n" + "-"*70)
    print("Iteration {} Results: Supervised={:.4f}, Autoencoder={:.4f}, VIME-Self={:.4f}, VIME={:.4f}".format(
      it+1, results[it, 0], results[it, 1], results[it, 2], results[it, 3]))
    print("-"*70)
  
  #%% Print results
  print("\n" + "#"*70)
  print("# FINAL RESULTS (Average over {} iterations)".format(args.iterations))
  print("#"*70)

  print('Supervised Performance, Model Name: ' + args.model_name +
        ', Avg Perf: ' + str(np.round(np.mean(results[:, 0]), 4)) +
        ', Std Perf: ' + str(np.round(np.std(results[:, 0]), 4)))

  print('Autoencoder Performance' +
        ', Avg Perf: ' + str(np.round(np.mean(results[:, 1]), 4)) +
        ', Std Perf: ' + str(np.round(np.std(results[:, 1]), 4)))

  print('VIME-Self Performance' +
        ', Avg Perf: ' + str(np.round(np.mean(results[:, 2]), 4)) +
        ', Std Perf: ' + str(np.round(np.std(results[:, 2]), 4)))

  print('VIME Performance' +
        ', Avg Perf: ' + str(np.round(np.mean(results[:, 3]), 4)) +
        ', Std Perf: ' + str(np.round(np.std(results[:, 3]), 4)))

  print("#"*70)
  
  
#%%  
if __name__ == '__main__':

  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--iterations',
      help='number of experiments iterations',
      default=10,
      type=int)
  parser.add_argument(
      '--model_name',
      choices=['logit','xgboost','mlp'],
      default='mlp',
      type=str)
  parser.add_argument(
      '--label_no',
      help='number of labeled data to be used',
      default=1000,
      type=int)
  parser.add_argument(
      '--p_m',
      help='corruption probability for self-supervised learning',
      default=0.3,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyper-parameter to control the weights of feature and mask losses',
      default=2.0,
      type=float)
  parser.add_argument(
      '--K',
      help='number of augmented samples',
      default=3,
      type=int)
  parser.add_argument(
      '--beta',
      help='hyperparameter to control supervised and unsupervised loss',
      default=1.0,
      type=float)
  parser.add_argument(
      '--label_data_rate',
      help='ratio of labeled data',
      default=0.1,
      type=float)
  parser.add_argument(
      '--dataset',
      choices=['mnist', 'synthetic'],
      help='dataset to use (mnist or synthetic)',
      default='synthetic',
      type=str)
  parser.add_argument(
      '--n_noise_features',
      help='number of irrelevant/noise features to add (synthetic data only)',
      default=0,
      type=int)
  parser.add_argument(
      '--seed',
      help='random seed for reproducibility',
      default=42,
      type=int)
  parser.add_argument(
      '--hidden_dim',
      help='hidden layer dimension for all models',
      default=100,
      type=int)
  parser.add_argument(
      '--patience',
      help='early stopping patience (epochs/iterations)',
      default=5,
      type=int)

  args = parser.parse_args()

  # Calls main function
  results = exp_main(args)