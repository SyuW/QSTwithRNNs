import torch
import numpy as np


def load_data(path, batchsize):
    data = np.loadtxt(path)
    data = torch.utils.data.DataLoader(data, batch_size=batchsize)
    return data

def load_observables(N):
    psi_N=np.loadtxt("data/psi_N="+str(N))
    energies=np.loadtxt("data/energies", skiprows=1)
    return psi_N[:,0], energies[int(N/2)-1,1]
