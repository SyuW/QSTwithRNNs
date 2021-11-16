"""
Recurrent neural network

Authors: Uzair Lakhani, Luc Andre Ouellet, Jefferson Pule Mendez
"""

import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim


class ConventionalRNN(nn.Module):

    def __init__(self):
        super(ConventionalRNN, self).__init__()

        hidden_units = 100
        random_seed = 1
        system_size = 10
        n_samples = 1000

        # hidden vector
        self.h_init = torch.zeros(hidden_units)
        # input spin

        self.gru_cell = nn.GRUCell(input_size=2, hidden_size=hidden_units, bias=True)  # n_h, 2 -> n_h
        self.linear = nn.Linear(in_features=hidden_units, out_features=2, bias=True)  # n_h -> 2
        self.softmax = nn.Softmax()  # 2 -> 2

        # input for RNN
        input_batch = torch.zeros(size=(n_samples,), ).long()
        self.sigma_0 = torch.nn.functional.one_hot(input_batch, classes=2)

        return

    def forward(self, spin, hidden_vec):
        new_hidden_vec = self.gru_cell(spin, hidden_vec)
        probabilities = self.softmax(new_hidden_vec)

        return new_hidden_vec, probabilities

    def get_data(self):
        return

    def backprop(self):
        return


if __name__ == "__main__":
    pass
