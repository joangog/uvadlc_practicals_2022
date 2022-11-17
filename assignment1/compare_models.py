################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-01
################################################################################
"""
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch

import matplotlib.pyplot as plt

import json

import torch
import torch.nn as nn
import torch.optim as optim
# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Run all hyperparameter configurations as requested

    # Hyperparameter Configurations
    kwargs_list_2_4 = []  # for Question 2.4.iii
    kwargs_list_2_5 = []  # for Question 2.5.vi
    lr_list = 10 ** np.linspace(-6, 2, 9)
    hidden_dims = [[128], [256, 128], [512, 256, 128]]
    for lr in lr_list:
        kwargs_list_2_4.append({'epochs': 10, 'lr': lr, 'hidden_dims': [128], 'batch_size': 128, 'use_batch_norm': False, 'data_dir': 'data/', 'seed': 42})
    for hidden_dim in hidden_dims:
        kwargs_list_2_5.append({'epochs': 20, 'lr': 0.1, 'hidden_dims': hidden_dim, 'batch_size': 128, 'use_batch_norm': True, 'data_dir': 'data/', 'seed': 42})

    # Initialize results dict
    results = {
        '2_4': {
            'kwargs_list': kwargs_list_2_4,
            'train_accuracies_list': [],
            'val_accuracies_list': [],
            'test_accuracy_list': [],
            'losses_list': []
        },
        '2_5': {
            'kwargs_list': kwargs_list_2_5,
            'train_accuracies_list': [],
            'val_accuracies_list': [],
            'test_accuracy_list': [],
            'losses_list': []
        }
    }

    # Train for every configuration in Question 2.4.iii
    for kwargs in kwargs_list_2_4:
        print()
        print(f'HYPERPARAMETERS: {kwargs}')
        print()
        # Train Model
        model, val_accuracies, test_accuracy, logging_info = train_mlp_pytorch.train(**kwargs)
        # Save model results
        results['2_4']['train_accuracies_list'].append(logging_info['train_accuracies'])
        results['2_4']['val_accuracies_list'].append(val_accuracies)
        results['2_4']['test_accuracy_list'].append(test_accuracy)
        results['2_4']['losses_list'].append(logging_info['losses'])

    # Train for every configuration in Question 2.5.vi
    for kwargs in kwargs_list_2_5:
        print()
        print(f'HYPERPARAMETERS: {kwargs}')
        print()
        # Train Model
        model, val_accuracies, test_accuracy, logging_info = train_mlp_pytorch.train(**kwargs)
        # Save model results
        results['2_5']['train_accuracies_list'].append(logging_info['train_accuracies'])
        results['2_5']['val_accuracies_list'].append(val_accuracies)
        results['2_5']['test_accuracy_list'].append(test_accuracy)
        results['2_5']['losses_list'].append(logging_info['losses'])

    json_object = json.dumps(results, indent=4)
    with open(results_filename, "a") as f:
        f.write(json_object)

    # TODO: Save all results in a file with the name 'results_filename'. This can e.g. by a json file

    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    with open(results_filename, "r") as f:
        results = json.load(f)

    lr_list = [kwargs['lr'] for kwargs in results['2_4']['kwargs_list']]

    # Loss plot (Question 2.4.iii)
    plt.title('Cross-Entropy Loss curve for PyTorch MLP')
    plt.plot(np.array(results['2_4']['losses_list']).T)
    plt.legend([f'lr = {lr}' for lr in lr_list])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Validation Accuracy per Learning Rate plot (Question 2.4.iii)
    plt.figure()
    plt.title('Validation Accuracy for different learning rates')
    plt.xscale('log')
    plt.plot(lr_list, [np.max(val_accuracies) for val_accuracies in results['2_4']['val_accuracies_list']])
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')

    # Training and Validation Accuracy plot (Question 2.5.vi)
    for i, kwargs in enumerate(results['2_5']['kwargs_list']):
        plt.figure()
        plt.title(f'Training Accuracy curve with hidden_dims = {kwargs["hidden_dims"]}')
        plt.plot(np.array(results['2_5']['train_accuracies_list'][i]).T)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.figure()
        plt.title(f'Validation Accuracy curve with hidden_dims = {kwargs["hidden_dims"]}')
        plt.plot(np.array(results['2_5']['val_accuracies_list'][i]).T)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

    plt.show()

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    FILENAME = 'results.json'
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)