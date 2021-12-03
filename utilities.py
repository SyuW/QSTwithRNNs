"""
Utility functions

Authors: Sam Yu, Jefferson Pule Mendez, Luc Andre Ouellet
"""

import torch
import numpy as np


def load_data(path, batchsize):
    """
    load the dataset for training
    :param path:
    :param batchsize:
    :return:
    """
    data = np.loadtxt(path)
    data = torch.utils.data.DataLoader(data, batch_size=batchsize)
    return data


def load_observables(N):
    """
    load observables from data files
    :param N:
    :return:
    """
    if N in [2, 4, 10]:
        psi_N = np.loadtxt("data/psi_N=" + str(N))
    else:
        psi_N = np.random.randn(2, 2)
    energies = np.loadtxt("data/energies", skiprows=1)
    
    energies_dict = dict()
    for i, n in enumerate(energies[:, 0]):
        energies_dict[n] = energies[i, 1]
    print(energies_dict)
    energy = energies_dict[N]

    return psi_N[:, 0], energy


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
    """
    convert the samples to binary
    :param samples:
    :return:
    """
    for i in range(samples.shape[1]):
        samples[:, samples.shape[1] - i - 1] *= 2 ** i
        samples_in_bin = torch.sum(samples, dim=1, keepdim=True)

    return samples_in_bin


def compute_fidelity(samples, probs, _gs_psi):
    """
    compute the fidelity of RNN ground state
    :param samples:
    :param probs:
    :param _gs_psi:
    :return:
    """
    probs = torch.prod(probs, dim=1, keepdim=True)

    samples_in_bin = transform_states_to_binary(samples)
    samples_and_probs = torch.cat([samples_in_bin, probs], dim=1)
    unique_samples = torch.unique(samples_in_bin)

    fidelity = 0
    rnn_psi_sigmas = []

    for sigma in unique_samples:

        sigma = int(sigma.numpy())
        GS_psi_sigma = _gs_psi[sigma]

        for sam_and_pr in samples_and_probs:
            if sigma == sam_and_pr[0]:
                rnn_psi_sigma = np.sqrt(sam_and_pr[1].numpy())
                break

        rnn_psi_sigmas.append([sigma, rnn_psi_sigma])
        fidelity += GS_psi_sigma * rnn_psi_sigma

    return fidelity ** 2, rnn_psi_sigmas

