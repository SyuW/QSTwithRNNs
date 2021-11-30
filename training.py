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
import pickle


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


def transform_states_to_binary(samples):
    for i in range(samples.shape[1]):
        samples[:, samples.shape[1] - i - 1] *= 2 ** i
        samples_in_bin = torch.sum(samples, dim=1, keepdim=True)

    return samples_in_bin


def Fidelity(samples, probs, GS_psi):
    #    print("############### Fidelity start #################")
    #    print("cond probs", probs)
    probs = torch.prod(probs, dim=1, keepdim=True)

    # Calculate PSI of the RNN

    #    print("samples",samples)
    #    print("probs",probs)
    #    print("GS_psi",GS_psi)

    samples_in_bin = transform_states_to_binary(samples)
    samples_and_probs = torch.cat([samples_in_bin, probs], dim=1)
    unique_samples = torch.unique(samples_in_bin)
    #    print("samples_and_probs",samples_and_probs)
    #    print("unique samples", unique_samples)

    fidelity = 0
    RNN_psi_sigmas = []

    for sigma in unique_samples:
        sigma = int(sigma.numpy())
        #        print(sigma)
        GS_psi_sigma = GS_psi[sigma]

        #        print("GS_psi_sigma",GS_psi_sigma)

        for sam_and_pr in samples_and_probs:

            if sigma == sam_and_pr[0]:
                RNN_psi_sigma = np.sqrt(sam_and_pr[1].numpy())
                #                print("RNN_psi_sigma",RNN_psi_sigma)
                break
        RNN_psi_sigmas.append([sigma, RNN_psi_sigma])
        fidelity += GS_psi_sigma * RNN_psi_sigma
    #        print("fidelity",fidelity)

    return fidelity ** 2, RNN_psi_sigmas


def train(model, data, results_path, num_epochs, display_epochs, learning_rate,
          truth_energy, truth_psi, verbose=True):
    """
    train the model

    :param truth_psi:
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
    infidelity_vals = []
    energy_diff_vals = []
    RNN_psi_sigmas_epochs = []

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

            # calculate percentage of samples with Sz non-zero
            samples, samples_probs = model.get_samples_and_probs(n_samples=1000, get_same_sample=False, verbose=False)
            nonzero_sz_percent = calculate_nonzero_sz_percent(samples)
            nonzero_sz_vals.append(nonzero_sz_percent)

            # calculate the energy difference
            rnn_energy_per_spin = model.calculate_xy_energy(samples) / model.num_spins
            energy_diff = torch.abs(rnn_energy_per_spin - truth_energy)
            energy_diff_vals.append(energy_diff)

            # calculate the fidelity if
            if int(model.num_spins) in [2, 4, 10]:
                fidelity, RNN_psi_sigmas = Fidelity(samples, samples_probs, truth_psi)
                infidelity_vals.append(1 - fidelity)
                RNN_psi_sigmas_epochs.append(RNN_psi_sigmas)
            else:
                fidelity = 0

        # use loss value for last batch of epoch for plot
        obj_vals.append(obj_val.item())

        # print out the epoch and loss value every display_epochs
        if (epoch + 1) % display_epochs == 0:
            print((f"Epoch [{epoch + 1}/{num_epochs}]"
                   f"\tLoss: {obj_val:.4f}"
                   f"\tInfidelity: {1 - fidelity:.4f}"
                   f"\tEnergy difference: {energy_diff:.4f}"))

    # Save PSIs:
    if int(model.num_spins) in [2, 4, 10]:
        if not model.symmetry:
            with open(save_path + f"/N={model.num_spins}" + f"psi_N={model.num_spins}_RNN.pkl", "wb") as file:  # Pickling
                pickle.dump(RNN_psi_sigmas_epochs, file)
        else:
            with open(save_path + f"/N={model.num_spins}" + f"psi_N={model.num_spins}_U(1).pkl", "wb") as file:  # Pickling
                pickle.dump(RNN_psi_sigmas_epochs, file)

    # save the arrays with loss, non-zero Sz, infidelity, energy differences
    loss_fname = f"loss_N_{model.num_spins}_symm_{model.symmetry}"
    sz_fname = f"sz_N_{model.num_spins}_symm_{model.symmetry}"
    energy_diff_fname = f"energy_diff_{model.num_spins}_symm_{model.symmetry}.npy"
    np.save(os.path.join(results_path, loss_fname), np.array(obj_vals))
    np.save(os.path.join(results_path, sz_fname), np.array(nonzero_sz_vals))
    np.save(os.path.join(results_path, energy_diff_fname), np.array(energy_diff_vals))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN for Quantum State Tomography")
    parser.add_argument("-json", default="params/params.json", help="input path to json file")
    parser.add_argument("-system_size", type=int, default=4, help="Size of our system. Default 10")
    parser.add_argument("-results_path", default="results", help="file path to results")
    args = parser.parse_args()

    # Load the model parameters
    with open(args.json, 'r') as f:
        params = json.load(f)

        lr = params['training']['learning rate']
        random_seed = params['training']['random seed']  # Where do we define the random seed
        epochs = params['training']['epochs']
        de = params['training']['display epochs']
        hidden_units = params['model']['hidden units']
        batch_size = params['data']['batch size']

    # make the directory to store results at
    save_path = os.path.join(args.results_path, f"N={args.system_size}")
    os.makedirs(save_path, exist_ok=True)

    # create the data loader
    data_loader = load_data(f"data/samples_N={args.system_size}_batch=1", params['data']['batch size'])
    gs_psi, dmrg_energy = load_observables(args.system_size)

    # initialize the model
    rnn = ConventionalRNN(hidden=hidden_units, system_size=args.system_size, seed=random_seed, symmetric=True)

    # start training
    import time

    start = time.time()
    train(rnn, data=data_loader, results_path=save_path, num_epochs=epochs, truth_energy=dmrg_energy,
          truth_psi=gs_psi, learning_rate=lr, display_epochs=de, verbose=False)

    print(f"Execution time: {time.time() - start} seconds")
