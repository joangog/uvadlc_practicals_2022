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
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        n_hidden_layers = len(n_hidden)

        in_features = [n_inputs,] + n_hidden # Get the number of inputs per  linear layer
        out_features = n_hidden + [n_classes,] # Get the number of outputs per hidden linear layer
        input_layer = [True,] + [False for i in range(0, n_hidden_layers + 1)]  # Set True only for the first hidden linear layer

        # Layer Modules
        self.linear_layers = [LinearModule(in_features[i], out_features[i], input_layer[i]) \
                              for i in range(n_hidden_layers + 1)]
        self.activation_layers = [ELUModule() for i in range(n_hidden_layers)] + [SoftMaxModule(),]
        self.loss = CrossEntropyModule()

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i].forward(x)
            x = self.activation_layers[i].forward(x)

        out = x

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        for i in range(len(self.linear_layers)-1, -1, -1):
            dout = self.activation_layers[i].backward(dout)
            dout = self.linear_layers[i].backward(dout)

        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        for layer in self.linear_layers:
            layer.clear_cache()
        for layer in self.activation_layers:
            layer.clear_cache

        #######################
        # END OF YOUR CODE    #
        #######################
