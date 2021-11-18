"""
Recurrent neural network

Authors: Uzair Lakhani, Luc Andre Ouellet, Jefferson Pule Mendez
"""

import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import json

from data import data_load


class ConventionalRNN(nn.Module):

    def __init__(self, hidden, system_size, seed):
        super(ConventionalRNN, self).__init__()

        # parameters
        self.hidden_units = hidden
        self.random_seed = seed
        self.num_spins = system_size

        # recurrent cell architecture
        self.gru_cell = nn.GRUCell(input_size=2, hidden_size=self.hidden_units, bias=True)  # n_h, 2 -> n_h
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=2, bias=True)  # n_h -> 2
        self.softmax = nn.Softmax(dim=1)  # 2 -> 2

    def forward(self, spin, hidden_vec):
        """
        Returns a new value for the hidden vector and a set of probabilities 

        :param spin: Tensor of size (batch_size, 1): batch of one-hotted spins
        :param hidden_vec: Tensor of size (batch_size, hidden_units) hidden vector
        """

        new_hidden_vec = self.gru_cell(spin, hidden_vec)
        linear_output = self.linear(new_hidden_vec)
        probabilities = self.softmax(linear_output)  # Probability of getting spin sigma_n+1

        return new_hidden_vec, probabilities

    @staticmethod
    def get_next_spin_and_prob(rnn_iteration, probabilities, data, training):
        """
        Function returns the new spin configuration (sigma_(n+1)), the one hot encoded version of it and the 
        probability of getting that specific output, given probabilities. 
        
        RNN_iteration    =  int  
                            number of spin that the RNN is iterating at the time

        data             =  pytorch tensor
                            The data set for training. (Useful when data is not required).

        training         =  boolean
                            Whether we are training using data or not. If false then just sample randomly.

        sample_sigma_n   =  Tensor of size (ns,1)
                            New set of samples 

        sigma_n_encoded  =  Tensor of size (ns, 2)
                            One hot encoded of sample_sigma_n

        conditional_prob =  Tensor of size (ns, 2)
                            Bernoulli distribution for sample_sigma_n on step RNN_iteration
        """

        n = rnn_iteration

        # training mode: use spin from dataset
        if training:
            # get the spin from dataset
            sample_sigma_n = data[:, n].long()

        # sampling mode: sample a new spin from probability distribution
        else:
            # sample the spin from Bernoulli distribution
            prob_of_getting_1 = probabilities[:, 1]
            sample_sigma_n = torch.bernoulli(prob_of_getting_1).long()  # Check dimensions

        # convert to one-hot encoding
        sigma_n_encoded = torch.nn.functional.one_hot(sample_sigma_n, num_classes=2).float()

        # compute the conditional probability
        y_n_sigma_n = probabilities * sigma_n_encoded
        conditional_prob = torch.sum(y_n_sigma_n, dim=1, keepdim=True)

        # reshape into a column vector, not one-hotted
        sample_sigma_n = torch.reshape(sample_sigma_n, shape=(sample_sigma_n.shape[0], 1))

        return sample_sigma_n, sigma_n_encoded, conditional_prob

    def train_or_sample(self, batch=[], n_samples=30, training=False, verbose=True):
        """
        Function sequentially performs num_spins iterations returning for each iteration
        a sample sigma_i and its conditional probability. The output are two tensors containing
        the probabilities of a full configuration sigma and the individual conditional probabilities.

        batch             =  pytorch tensor
                             The data set for training. It will be equal to 1 if not data is provided
                             (Useful when data is not required).

        n_samples         =  number of spin configuration samples to generate

        training          =  boolean. 
                             Whether we are training using data or not. If false then just sample randomly.

        probabilities     =  Tensor of size (batch_size, size_of_system)
                             Probability of each configuration sigma 

        sampled_spins     =  Tensor of size (batch_size, size_of_system)
                             Configuration sigma
        """

        # infer batch size from training dataset
        if training:
            batch_size = batch.shape[0]
            print(f"working with ns={batch_size}")
        # else, if sampling, use n_samples parameter
        else:
            batch_size = n_samples

        # initialize probabilities and sampled spins to zero
        probabilities = torch.zeros(size=(batch_size, 1))
        sampled_spins = torch.zeros(size=(batch_size, 1))

        # initialize the hidden vector and the initial spin for RNN
        input_batch = torch.zeros(size=(batch_size,)).long()
        h_n = torch.zeros(size=(batch_size, self.hidden_units)).float()
        sigma_n = torch.nn.functional.one_hot(input_batch, num_classes=2).float()

        # iterate over spin configuration
        for n in range(self.num_spins):

            if verbose:
                print(f"Spin No. {n}")

            # update the hidden vector and obtain probability distribution for next spin
            h_n, prob = self.forward(sigma_n, h_n)

            # get the next spin
            sampled_spin, sigma_n, conditional_prob = self.get_next_spin_and_prob(n, prob, batch, training)

            # if initial recurrent cell, simply add since variables were initialized to zero
            if n == 0:
                probabilities = torch.add(probabilities, conditional_prob)
                sampled_spins = torch.add(sampled_spins, sampled_spin)

            # else, concatenate
            else:
                probabilities = torch.cat([probabilities, conditional_prob], dim=1)
                sampled_spins = torch.cat([sampled_spins, sampled_spin], dim=1)

        if verbose:
            print(f"Training mode: {training}")
            print(f"Probabilities: {probabilities}")
            print(f"Generated sample: {sampled_spins}")
            print(f"Input data: {batch}")

        return sampled_spins, probabilities


if __name__ == "__main__":
    hidden_units = 100
    random_seed = 1
    sys_size = 4

    model = ConventionalRNN(hidden_units, sys_size, random_seed)
    test = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 1, 1],
                         [1, 1, 1, 1], [0, 1, 1, 1]])  # DATA SHOULD BE OUR BATCHES IN TRAINING
    P, S = model.train_or_sample(batch=test, training=True)  # INCLUDE DATA WHEN TRAINING TRUE
    P, S = model.train_or_sample(n_samples=30, training=False)  # DO NOT INCLUDE DATA WHEN TRAINING NOT TRUE
