import torch
import numpy as np


def data_load(path, batchsize):
    data = np.loadtxt(path)
    data = torch.utils.data.DataLoader(data, batch_size=batchsize)
    return data