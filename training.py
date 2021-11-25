"""
Training procedure
Authors: Sam Yu, Jefferson Pule Mendez, Luc Andre Ouellet
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from RNN import ConventionalRNN
from data import load_data, load_observables

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


def calculate_nonzero_sz_percent(samples):
    """
    Compute the percentage of samples with non-zero net magnetization
    :param samples:
    :return:
    """
    # convert back to eigenvalues for S_z
    num_sz_not_zero = torch.nonzero(torch.sum(samples - 1 / 2, dim=1)).shape[0]
    total = samples.shape[0]
    nonzero_sz = num_sz_not_zero / total

    return nonzero_sz


def train(model, data, results_path, num_epochs, display_epochs, learning_rate, truth_energy, verbose=True):
    """
    train the model

    :param truth_energy:
    :param verbose:
    :param learning_rate:
    :param display_epochs:
    :param num_epochs:
    :param results_path:
    :param data:
    :param model:
    :return:
    """

    # defining the optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    obj_vals = []
    nonzero_sz_vals = []
    # start the training
    for epoch in range(num_epochs):

        for batch in data:
            # clear gradients
            optimizer.zero_grad()

            # calculate probabilities
            sampled_spins, probs = model.get_samples_and_probs(batch=batch, get_same_sample=True, verbose=False)
            config_probabilities = torch.prod(probs, dim=1, keepdim=True)

            # compute the loss
            obj_val = negative_log_loss(config_probabilities)

            # calculate gradients and update parameters
            obj_val.backward()
            optimizer.step()

        # sample from RNN probability distribution at the end of each epoch
        with torch.no_grad():
            samples, _ = model.get_samples_and_probs(n_samples=1000, get_same_sample=False, verbose=False)
            # calculate percentage of samples with Sz non-zero
            nonzero_sz_percent = calculate_nonzero_sz_percent(samples)
            nonzero_sz_vals.append(nonzero_sz_percent)
            # calculate the energy difference
            rnn_energy_per_spin = model.calculate_xy_energy(samples) / model.num_spins
            energy_diff = torch.abs(rnn_energy_per_spin - truth_energy)

        # use loss value for last batch of epoch for plot
        obj_vals.append(obj_val.item())

        # print out the epoch and loss value every display_epochs
        if (epoch + 1) % display_epochs == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}\tLoss: {obj_val:.4f}]\tEnergy diff: {energy_diff:.4f}")
            print(f"Energy: {rnn_energy_per_spin}")

    # create all the plots
    with plt.ioff():
        fig, ax = plt.subplots()

    # loss plot
    ax.plot(range(num_epochs), obj_vals, color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log Loss")
    ax.set_title(f"Loss vs epoch for N={batch.shape[1]}")
    fig.savefig(os.path.join(results_path, "loss_plot.png"))
    ax.cla()

    # non-zero S_z samples plot
    ax.plot(range(num_epochs), nonzero_sz_vals, color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Percentage of samples with $S_z \neq 0$")
    ax.set_title(r"Fraction of samples with $S_z \neq 0$ " + f"for N={batch.shape[1]}")
    fig.savefig(os.path.join(results_path, "nonzero_sz_plot.png"))
    ax.cla()

    # close all active figures
    plt.close()

    # save the arrays with loss values and S_z non-zero
    np.save(os.path.join(results_path, f"loss_N_{batch.shape[1]}.npy"), np.array(obj_vals))
    np.save(os.path.join(results_path, f"sz_N_{batch.shape[1]}.npy"), np.array(nonzero_sz_vals))

    # write to report file
    with open(os.path.join(results_path, "report.txt"), 'w') as report_file:
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
        de = params['optim']['display epochs']
        hidden_units = params['model']['hidden units']
        batch_size = params['data']['batch size']
        n_samples = 10

    # make the directory to store results at
    save_path = os.path.join(args.results_path, f"N={args.system_size}")
    os.makedirs(save_path, exist_ok=True)

    # create the data loader
    data_loader = load_data(f"data/samples_N={args.system_size}_batch=1", params['data']['batch size'])
    gs_psi, dmrg_energy = load_observables(args.system_size)

    # initialize the model
    rnn = ConventionalRNN(hidden=hidden_units, system_size=args.system_size, seed=random_seed)

    # start training
    import time

    start = time.time()
    train(rnn, data=data_loader, results_path=save_path, num_epochs=epochs,
          truth_energy=dmrg_energy, learning_rate=lr, display_epochs=de, verbose=False)

    print(f"Execution time: {time.time() - start} seconds")
