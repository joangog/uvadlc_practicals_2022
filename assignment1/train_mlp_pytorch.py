################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2021-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    batch_size, n_classes = predictions.shape

    conf_mat = torch.zeros((n_classes, n_classes))

    # Convert class probabilities into class labels (0, 1, 2, 3, etc.)
    predictions = torch.argmax(predictions, dim=1)

    # Build confusion matrix
    for i in range(batch_size):
        pred_class = predictions[i]
        target_class = targets[i]
        conf_mat[pred_class, target_class] += 1

    #######################
    # END OF YOUR CODE    #
    #######################

    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    accuracy = torch.sum(torch.diag(confusion_matrix)) / torch.sum(confusion_matrix)
    precision = torch.diag(confusion_matrix) / (torch.sum(confusion_matrix, axis=1))
    recall = torch.diag(confusion_matrix) / (torch.sum(confusion_matrix, axis=0))
    f1_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_beta': f1_beta
    }

    #######################
    # END OF YOUR CODE    #
    #######################

    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    if type(data_loader.dataset).__name__ == 'Subset':  # For train and val dataset
        n_features = np.prod(data_loader.dataset.dataset.data.shape[1:])
        n_classes = len(data_loader.dataset.dataset.classes)
    else:  # For test dataset
        n_features = np.prod(data_loader.dataset.data.shape[1:])
        n_classes = len(data_loader.dataset.classes)

    conf_matrix = torch.zeros((n_classes, n_classes))

    for inputs, targets in data_loader:  # For every batch

        # Vectorize input samples
        n_inputs = inputs.shape[0]
        inputs = inputs.reshape((n_inputs, n_features))

        out = model.forward(inputs)

        conf_matrix += confusion_matrix(out, targets)

    metrics = confusion_matrix_to_metrics(conf_matrix)
    metrics['conf_matrix'] = conf_matrix

    #######################
    # END OF YOUR CODE    #
    #######################

    return metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set default device (Don't forget to send your data and layers to the device!)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    train_dataset = cifar10_loader['train']
    val_dataset = cifar10_loader['validation']
    test_dataset = cifar10_loader['test']

    n_features = np.prod(train_dataset.dataset.dataset.data.shape[1:])
    n_classes = len(train_dataset.dataset.dataset.classes)

    # TODO: Initialize model, optimizer and loss module
    model = MLP(n_inputs=n_features, n_hidden=hidden_dims, n_classes=n_classes, use_batch_norm=use_batch_norm).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # TODO: Training loop including validation

    train_accuracies = []  # Train accuracies for every epoch
    val_accuracies = []  # Validation accuracies for every epoch
    losses = []  # Loss for every epoch

    model_checkpoints = []  # Stores model checkpoints of every epoch

    # Train model
    print()
    print(f'Training for {epochs} epochs...')

    for epoch in range(epochs):  # For every epoch
        epoch_losses = []  # Loss for every batch in this epoch
        conf_matrix = torch.zeros((n_classes, n_classes))

        for inputs, targets in train_dataset:  # For every batch
            # Send data to GPU
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Vectorize input samples
            n_inputs = inputs.shape[0]
            inputs = inputs.reshape((n_inputs, n_features))

            # Forward step
            out = model.forward(inputs)
            loss = model.loss(out, targets)

            epoch_losses.append(loss.cpu().detach().item())
            conf_matrix += confusion_matrix(out, targets)

            # Backward step
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Clear gradients
            model.zero_grad()

        # Save model checkpoint
        model_checkpoints.append(model.state_dict())

        # Save epoch loss and metrics
        epoch_loss = np.mean(epoch_losses)  # Mean loss of epoch
        losses.append(epoch_loss)
        train_metrics = confusion_matrix_to_metrics(conf_matrix)
        train_accuracies.append(train_metrics['accuracy'].item())

        # Validate model in each epoch
        val_metrics = evaluate_model(model, val_dataset, n_classes)
        val_accuracies.append(val_metrics['accuracy'].item())
        print(f'   Epoch {epoch}:')
        print(f'      Loss: {round(epoch_loss, 2)}')
        print(f'      Accuracy: {round(val_accuracies[-1], 2)}')

    # Revert model to epoch with best model
    best_epoch = np.argmax(val_accuracies)
    model.load_state_dict(model_checkpoints[best_epoch])
    print(f'Best Epoch: Epoch {best_epoch}')
    print(f'      Loss: {round(losses[best_epoch], 2)}')
    print(f'      Accuracy: {round(val_accuracies[best_epoch], 2)}')
    print()

    # TODO: Test best model
    print('Testing...')
    test_metrics = evaluate_model(model, test_dataset, n_classes)
    test_accuracy = test_metrics['accuracy'].item()
    print(f'   Accuracy: {round(test_accuracy, 2)}')
    print()

    # TODO: Add any information you might want to save for plotting
    logging_info = {
        'losses': losses,
        'conf_matrix': test_metrics['conf_matrix'],
        'f1_beta': test_metrics['f1_beta'],
        'train_accuracies': train_accuracies
    }

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)

    # Feel free to add any additional functions, such as plotting of the loss curve here

    # Test Confusion Matrix and F1-scores for each class (for Question 2.3.ii)
    # Note 1: To see the values of the two variables below we used the PyCharm debugger:
    conf_matrix = logging_info['conf_matrix'].numpy().astype(int)
    f1_beta = logging_info['f1_beta'].numpy()

    plt.figure()
    plt.title('Cross-Entropy Loss curve for PyTorch MLP')
    plt.plot(logging_info['losses'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.figure()
    plt.title('Validation Accuracy curve for PyTorch MLP')
    plt.plot(val_accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()
