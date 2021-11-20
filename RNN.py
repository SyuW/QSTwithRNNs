"""
Recurrent neural network

Authors: Uzair Lakhani, Luc Andre Ouellet, Jefferson Pule Mendez
"""

import argparse

import torch
import torch.nn as nn
import json


class ConventionalRNN(nn.Module):

    def __init__(self, hidden, system_size, seed, symmetric=False):
        super(ConventionalRNN, self).__init__()

        # parameters
        self.hidden_units = hidden
        self.random_seed = seed
        self.num_spins = system_size

        # whether to impose U(1) symmetry on RNN
        self.symmetry = symmetric

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
        
        :param probabilities: tensor. probability distribution for sampling spin
        :param rnn_iteration: int. iteration of RNN
        :param data: tensor. the training dataset (Useful when data is not required)
        :param training: boolean. whether we are training using data or not. If false then just sample randomly
        :return: sample_sigma_n, sigma_n_encoded, conditional_prob
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

    def train_or_sample(self, batch=[], n_samples=30, training=False, verbose=False):
        """
        Function sequentially performs num_spins iterations returning for each iteration
        a sample sigma_i and its conditional probability. The output are two tensors containing
        the sampled spins and the individual conditional probabilities.

        :param batch: pytorch tensor. the data set for training. (useful when data is not required).
        :param n_samples: int. number of spin configuration samples to generate.
        :param training: boolean. whether we are training using data or not. if false then just sample randomly.
        :param verbose: boolean. verbosity level.

        :return: sampled_spins, probabilities
        """

        # infer batch size from training dataset
        if training:
            batch_size = batch.shape[0]
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

            # impose symmetry
            if self.symmetry:
                if n != 0:
                    prob = self.enforce_symmetry(prob, sampled_spins, self.num_spins)

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
            print(f"Symmetry: {self.symmetry}")
            print(f"Training mode: {training}")
            print(f"Probabilities: {probabilities}")
            print(f"Generated sample: {sampled_spins}")
            print(f"Input data: {batch}")

        return sampled_spins, probabilities

    @staticmethod
    def enforce_symmetry(prob, sampled_spins, num_spins):

        N_sampled_spins = sampled_spins.size()[1]
        N_pos = torch.sum(sampled_spins, dim=1, keepdim=True)
        N_neg = N_sampled_spins - N_pos
        N_half = num_spins / 2
        heaviside = torch.where(torch.cat([N_neg, N_pos], dim=1) >= N_half, 0, 1)
        return (prob * heaviside) / (torch.sum(prob * heaviside, dim=1, keepdim=True))


if __name__ == "__main__":
    hidden_units = 100
    random_seed = 1
    sys_size = 4

    model = ConventionalRNN(hidden_units, sys_size, random_seed, symmetric=True)
    test = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1],
                         [0, 0, 1, 1]])  # DATA SHOULD BE OUR BATCHES IN TRAINING
    p, s = model.train_or_sample(batch=test, training=True,
                                 verbose=True)  # INCLUDE DATA WHEN TRAINING TRUE
    model = ConventionalRNN(hidden_units, sys_size, random_seed, symmetric=False)
    p, s = model.train_or_sample(batch=test, training=True,
                                 verbose=True)  # INCLUDE DATA WHEN TRAINING TRUE
    p, s = model.train_or_sample(n_samples=30, training=False,
                                 verbose=False)  # DO NOT INCLUDE DATA WHEN TRAINING NOT TRUE
