"""
Training procedure

Authors: Sam Yu
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from RNN import ConventionalRNN
from data import data_load

import torch
import torch.optim as optim
import json


def negative_log_loss(inputs):
    """
    Compute the negative log loss

    :param inputs: tensor. raw probabilities from RNN model
    :return:
    """

    # use negative log likelihood for the loss function
    offset = 1e-7
    loss_val = -torch.log(inputs + offset).mean()

    return loss_val


def train(model, data, **kwargs):
    """
    train the model

    :param data:
    :param model:
    :param kwargs:
    :return:
    """

    # hyperparameters
    learning_rate = 0.01
    display_epochs = 10
    num_epochs = 200

    # defining the optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    obj_vals = []
    obj_vals_plot = []
    # start the training
    for epoch in range(num_epochs):

        for batch in data:

            # clear gradients
            optimizer.zero_grad()

            # calculate probabilities
            _, probs = model.train_or_sample(batch=batch, training=True, verbose=False)
            config_probabilities = torch.prod(probs, dim=1, keepdim=True)

            # compute the loss
            obj_val = negative_log_loss(config_probabilities)
            obj_vals.append(obj_val.item())

            # calculate gradients and update parameters
            obj_val.backward()
            optimizer.step()

        if kwargs["verbose"]:
            print(probs)
            print(config_probabilities)
            pass

        # print out the epoch and loss value every display_epochs
        if (epoch + 1) % display_epochs == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}\tLoss: {obj_val:.4f}]")
            obj_vals_plot.append(obj_val.item())

    with plt.ioff():
        fig, ax = plt.subplots()

    ax.plot(range(num_epochs), obj_vals_plot)
    ax.scatter(range(num_epochs), obj_vals_plot)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log Loss")
    ax.set_title("Loss during training")

    fig.savefig("results/loss_plot.png")

    # write to report file
    with open('results/report.txt', 'w') as report_file:
        report_file.write("-" * 50 + "\nBegin Training Report\n" + "-" * 50 + "\n")
        for epoch in range(num_epochs):
            report_file.write(f'Epoch [{epoch + 1}/{num_epochs}]\tLoss: {obj_vals[epoch]:.4f}\n')
        report_file.write("-" * 50 + "\nEnd Training Report\n" + "-" * 50 + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train RNN for Quantum State Tomography")

    parser.add_argument("-json", default="params/params.json", help="input path to json file")
    parser.add_argument("-system_size", type=int, default=4, help="Size of our system. Default 10")
    parser.add_argument("-results_path", default="results", help="file path to results")
    args = parser.parse_args()

    # Load the model parameters
    with open(args.json, 'r') as f:
        params = json.load(f)

        lr = params['optim']['learning rate']
        random_seed = params['optim']['random seed']  # Where do we define the random seed
        epochs = params['optim']['epochs']
        hidden_units = params['model']['hidden units']
        batch_size = params['data']['batch size']
        n_samples = 10

    # create the data loader
    data_loader = data_load(f"data/samples_N={args.system_size}_batch=1", params['data']['batch size'])

    # initialize the model
    rnn = ConventionalRNN(hidden=hidden_units, system_size=args.system_size, seed=random_seed)

    # start training
    import time
    start = time.time()
    train(rnn, data=data_loader, verbose=False)
    print(f"Execution time: {time.time() - start} seconds")
