"""
Training procedure

Authors: Sam Yu, Jefferson Pule Mendez, Luc Andre Ouellet
"""

import os
import pickle

import torch
import torch.optim as optim
import numpy as np

from utilities import calculate_nonzero_sz_percent, compute_fidelity


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


def train(model, data, results_path, num_epochs, display_epochs, learning_rate,
          truth_energy, truth_psi):
    """
    train the model

    :param truth_psi:
    :param truth_energy:
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
    infidelity_vals = []
    energy_diff_vals = []
    RNN_psi_sigmas_epochs = []

    # start the training
    for epoch in range(num_epochs):

        for batch in data:
            # clear gradients
            optimizer.zero_grad()

            # calculate probabilities
            sampled_spins, probs = model.get_samples_and_probs(batch=batch, get_same_sample=True)
            config_probabilities = torch.prod(probs, dim=1, keepdim=True)

            # compute the loss
            obj_val = negative_log_loss(config_probabilities)

            # calculate gradients and update parameters
            obj_val.backward()
            optimizer.step()

        # sample from RNN probability distribution at the end of each epoch
        with torch.no_grad():

            # calculate percentage of samples with Sz non-zero
            samples, samples_probs = model.get_samples_and_probs(n_samples=1000, get_same_sample=False)
            nonzero_sz_percent = calculate_nonzero_sz_percent(samples)
            nonzero_sz_vals.append(nonzero_sz_percent)

            # calculate the energy difference
            rnn_energy_per_spin = model.calculate_xy_energy(samples) / model.num_spins
            energy_diff = torch.abs(rnn_energy_per_spin - truth_energy)
            energy_diff_vals.append(energy_diff)

            # calculate the fidelity if system
            if int(model.num_spins) in [2, 4, 10]:
                fidelity, RNN_psi_sigmas = compute_fidelity(samples, samples_probs, truth_psi)
                infidelity_vals.append(1 - fidelity)
                RNN_psi_sigmas_epochs.append(RNN_psi_sigmas)
            else:
                fidelity = None

        # use loss value for last batch of epoch for plot
        obj_vals.append(obj_val.item())

        # print out the epoch and loss value every display_epochs
        if (epoch + 1) % display_epochs == 0:
            if fidelity is not None:
                print((f"Epoch [{epoch + 1}/{num_epochs}]"
                       f"\tLoss: {obj_val:.4f}"
                       f"\tInfidelity: {1 - fidelity:.4f}"
                       f"\tEnergy difference: {energy_diff:.4f}"))
            else:
                print((f"Epoch [{epoch + 1}/{num_epochs}]"
                       f"\tLoss: {obj_val:.4f}"
                       f"\tEnergy difference: {energy_diff:.4f}"))

    # Save psi's:
    if int(model.num_spins) in [2, 4, 10]:
        if not model.symmetry:
            with open(os.path.join(results_path, f"psi_N={model.num_spins}_RNN.pkl"), "wb") as file:  # Pickling
                pickle.dump(RNN_psi_sigmas_epochs, file)
        else:
            with open(os.path.join(results_path, f"psi_N={model.num_spins}_U(1).pkl"), "wb") as file:  # Pickling
                pickle.dump(RNN_psi_sigmas_epochs, file)

    # save the arrays with loss, non-zero Sz, infidelity, energy differences
    loss_fname = f"loss_N_{model.num_spins}_symm_{model.symmetry}"
    sz_fname = f"sz_N_{model.num_spins}_symm_{model.symmetry}"
    energy_diff_fname = f"energy_diff_{model.num_spins}_symm_{model.symmetry}.npy"
    np.save(os.path.join(results_path, loss_fname), np.array(obj_vals))
    np.save(os.path.join(results_path, sz_fname), np.array(nonzero_sz_vals))
    np.save(os.path.join(results_path, energy_diff_fname), np.array(energy_diff_vals))

    # save the infidelity values only for the following system sizes
    if int(model.num_spins) in [2, 4, 10]:
        infidelity_fname = f"infidelity_N_{model.num_spins}_symm_{model.symmetry}.npy"
        np.save(os.path.join(results_path, infidelity_fname), np.array(infidelity_vals))

    # write to report file
    with open(os.path.join(results_path, "report.txt"), 'w') as report_file:
        report_file.write("-" * 90 + "\nBegin Training Report\n" + "-" * 90 + "\n")
        for epoch in range(num_epochs):
            if int(model.num_spins) in [2, 4, 10]:
                entry = (f"Epoch [{epoch + 1}/{num_epochs}]"
                         f"\tLoss: {obj_vals[epoch]:.4f}"
                         f"\tInfidelity: {infidelity_vals[epoch]:.4f}"
                         f"\tEnergy difference: {energy_diff_vals[epoch]:.4f}\n")
            else:
                entry = (f"Epoch [{epoch + 1}/{num_epochs}]"
                         f"\tLoss: {obj_vals[epoch]:.4f}"
                         f"\tEnergy difference: {energy_diff_vals[epoch]:.4f}\n")
            report_file.write(entry)
        report_file.write("-" * 90 + "\nEnd Training Report\n" + "-" * 90 + "\n")
