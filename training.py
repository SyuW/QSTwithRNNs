"""
Training procedure

Authors: Sam Yu
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from RNN import ConventionalRNN

import torch
import torch.nn as nn
import torch.optim as optim


def calculate_loss(input, targets):
    """

    :param input: raw probabilities from RNN model
    :param targets:
    :return:
    """

    # use negative log likelihood for the loss function
    loss_fn = nn.NLLLoss()
    loss_val = loss_fn(input, targets)

    return loss_val


def train(model, dataset, **kwargs):
    """

    :param model:
    :param dataset:
    :param kwargs:
    :return:
    """

    # hyperparameters
    learning_rate = 0.001
    display_epochs = 10
    num_epochs = 100
    batch_size = 50

    # defining the optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # start the training
    for epoch in range(num_epochs):

        # clear gradients
        optimizer.zero_grad()

        predictions = ConventionalRNN()

        # compute the loss
        obj_val = calculate_loss(predictions, targets=)

        # calculate gradients and update parameters
        loss.backward()
        optimizer.step()

        if (num_epochs + 1) % display_epochs:
            print(f"Epoch [{epoch+1}/{num_epochs}\tLoss: {obj_val:.4f}]")

    with open('results/report.txt', 'w') as report_file:
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(help="Train RNN for Quantum State Tomography")


