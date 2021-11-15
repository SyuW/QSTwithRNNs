"""
Training procedure

Authors: Sam Yu
"""

import numpy as np
import matplotlib.pyplot as plt

from RNN import ConventionalRNN

import torch
import torch.nn as nn
import torch.optim as optim


def train(model):

    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    pass
