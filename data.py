import torch
import numpy as np


def load_data(path, batchsize):
    data = np.loadtxt(path)
    data = torch.utils.data.DataLoader(data, batch_size=batchsize)
    return data


def load_observables(N):
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
