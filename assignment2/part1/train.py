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
# Date Created: 2022-11-14
################################################################################

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models

# My imports
from tqdm.auto import tqdm  # For progress bar

from cifar100_utils import get_train_validation_set, get_test_set, add_augmentation


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Randomly initialize and modify the model's last layer for CIFAR100.
    model_params = [param for param in model.parameters()]
    for i, param in enumerate(model_params):
        if i != len(model_params) - 1:  # If all other layers except last one
            param.requires_grad = False  # Freeze layer
        else:  # If last layer
            if type(model).__name__ == 'ResNet':  # Each model has a different way to access the last layer
                # Modify outputs
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                # Randomize parameters
                nn.init.normal_(model.fc.weight, mean=0, std=0.01)
                nn.init.zeros_(model.fc.bias)
            else:
                # Modify outputs
                model.fc = torch.nn.Linear(model.classifier._modules[i].in_features, num_classes)
                # Randomize parameters
                nn.init.normal_(model.classifier._modules[i].weight, mean=0, std=0.01)
                nn.init.zeros_(model.classifier._modules[i].bias)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_set, val_set = get_train_validation_set(data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with validation after each epoch. Save the best model.
    val_accuracies = []  # Validation accuracies for every epoch
    losses = []  # Loss for every epoch

    model_checkpoints = []  # Stores model checkpoints of every epoch


    # Train model

    print()
    print(f'*** TRAINING ***')
    print()

    # Send model to device
    model.to(device)

    model.train()

    for epoch in range(epochs):  # For every epoch
        print(f'Epoch {epoch}:')
        epoch_losses = []  # Loss for every batch in this epoch

        for inputs, targets in tqdm(train_loader):  # For every batch

            # Apply augmentation
            transform_list = []
            if augmentation_name:
                transform_list = add_augmentation(augmentation_name, transform_list)
                inputs = transform_list(inputs)

            # Send data to GPU
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward step
            out = model.forward(inputs)
            loss = F.cross_entropy(out, targets)

            epoch_losses.append(loss.cpu().detach().item())

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

        # Validate model in each epoch
        val_accuracy = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        print(f'      - Loss: {round(epoch_loss, 2)}')
        print(f'      - Accuracy: {round(val_accuracy * 100, 2)}%')
        print()

    # Load the best model on val accuracy and return it.
    best_epoch = np.argmax(val_accuracies)
    model.load_state_dict(model_checkpoints[best_epoch])
    print(f'Best Epoch: Epoch {best_epoch}')
    print(f'      - Loss: {round(losses[best_epoch], 2)}')
    print(f'      - Accuracy: {round(val_accuracies[best_epoch] * 100, 2)}%')
    print()

    # Save best model state dictionary
    torch.save(model_checkpoints[best_epoch], f'{checkpoint_name}.pt')

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Send model to device
    model.to(device)

    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().

    accuracy_list = []

    for inputs, targets in data_loader:  # For every batch

        # Send data to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            out = model.forward(inputs)
            out = torch.argmax(out, axis=1)  # Convert from probabilities to 1-hot

        accuracy_list.append((sum(out == targets) / len(targets)).detach().cpu().item())

    accuracy = np.mean(accuracy_list)

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    start = time.process_time()

    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = get_model()

    # Get the augmentation to use
    pass

    # Train the model
    model = train_model(model, lr, batch_size, epochs, data_dir, f'best_model_{time.time()}', device, augmentation_name)


    # Evaluate the model on the test set

    print()
    print('*** TESTING ***')
    print()

    test_set = get_test_set(data_dir)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    accuracy = evaluate_model(model, test_loader, device)

    print(f'      - Accuracy: {round(accuracy * 100, 2)}%')
    print()

    end = time.process_time()

    print(f'Time elapsed: {int((end - start) / 60)} minutes')

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
