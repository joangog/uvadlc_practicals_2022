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

"""Defines various kinds of visual-prompting modules for images."""
import random

import torch
import torch.nn as nn
import numpy as np


class PadPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Define the padding as variables self.pad_left, self.pad_right, self.pad_up, self.pad_down

        # Hints:
        # - Each of these are parameters that we need to learn. So how would you define them in torch?
        # - See Fig 2(c) in the assignment to get a sense of how each of these should look like.
        # - Shape of self.pad_up and self.pad_down should be (1, 3, pad_size, image_size)
        # - See Fig 2.(g)/(h) and think about the shape of self.pad_left and self.pad_right

        self.args = args

        self.pad_size = pad_size
        self.image_size = image_size

        self.pad_left = nn.Parameter(torch.randn((1, 3, image_size - 2 * pad_size, pad_size)))
        self.pad_right = nn.Parameter(torch.randn((1, 3, image_size - 2 * pad_size, pad_size)))
        self.pad_up = nn.Parameter(torch.randn((1, 3, pad_size, image_size)))
        self.pad_down = nn.Parameter(torch.randn((1, 3, pad_size, image_size)))

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, add the prompt as a padding to the image.

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        batch_size = x.shape[0]

        x_new = x.clone().to()

        x_new[:, :, self.pad_size:self.image_size-self.pad_size,:self.pad_size] = self.pad_left.repeat(batch_size, 1, 1, 1) # Repeats the padding for every image in the batch
        x_new[:, :, self.pad_size:self.image_size-self.pad_size, self.image_size-self.pad_size:self.image_size] = self.pad_right.repeat(batch_size, 1, 1, 1)
        x_new[:, :, :self.pad_size, :self.image_size] = self.pad_up.repeat(batch_size, 1, 1, 1)
        x_new[:, :, self.image_size-self.pad_size:self.image_size, :self.image_size] = self.pad_down.repeat(batch_size, 1, 1, 1)


        # Apply checkerboard effect

        h, w = x.shape[2], x.shape[3]  # Shape of checkerboard mask matrix
        s = self.args.square_size  # Check size

        if s != 0:  # If to apply checkerboard effect

            # Create checkerboard mask of 1s and 0s
            # Note: Method copied from https://stackoverflow.com/questions/72874737/how-to-make-a-checkerboard-in-pytorch
            indices = torch.stack(torch.meshgrid(torch.arange(h//s), torch.arange(w//s))).to(x.device)
            checkerboard = (indices.sum(dim=0) % 2).repeat_interleave(s, 0).repeat_interleave(s, 1)

            # Multiplies the old x with the checkerboard mask and the new x with the inverted checkerboard max
            x = x * checkerboard + (torch.abs(checkerboard-1)) * x_new

        else:  # If not to apply then just return the padded x
            x = x_new

        return x

        #######################
        # END OF YOUR CODE    #
        #######################


class FixedPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a fixed patch over an image.
    For refernece, this prompt should look like Fig 2(a) in the PDF.
    """
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can define as self.patch) of size [prompt_size, prompt_size]
        # that is placed at the top-left corner of the image.

        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn

        self.prompt_size = args.prompt_size
        self.image_size = args.image_size

        self.patch = nn.Parameter(torch.randn((1, 3, self.prompt_size, self.prompt_size)))

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, place the patch at the top-left

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        batch_size = x.shape[0]

        x[:, :, :self.prompt_size, :self.prompt_size] = self.patch.repeat(batch_size, 1, 1, 1)

        return x

        #######################
        # END OF YOUR CODE    #
        #######################


class RandomPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a random patch in the image.
    For refernece, this prompt should look like Fig 2(b) in the PDF.
    """
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can be defined as self.patch) of size [prompt_size, prompt_size]
        # that is located at the top-left corner of the image.

        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn

        self.prompt_size = args.prompt_size
        self.image_size = args.image_size

        self.patch = nn.Parameter(torch.randn((1, 3, self.prompt_size, self.prompt_size)))

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, place the patch at the top-left

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - Note that, here, you need to place the patch at a random location
        #   and not in the top-left corner.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        batch_size = x.shape[0]

        # Generate random indices for patch position
        i = random.randint(0, self.image_size - self.prompt_size)
        j = random.randint(0, self.image_size - self.prompt_size)

        x[:, :, i:i+self.prompt_size, j:j+self.prompt_size] = self.patch.repeat(batch_size, 1, 1, 1)

        return x

        #######################
        # END OF YOUR CODE    #
        #######################

