# Quantum State Tomography with U(1)-symmetric RNNs

Final project for Phys 449: Machine Learning in Physics taken during Fall 2021 with Pooya Ronagh.

Reproducing the results of a recently published paper https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.012401, we built a recurrent neural network that can reconstruct the spin-1/2 XY model's ground state wavefunction after training on density matrix renormalization group (DMRG) data. The spin-1/2 XY model has the following Hamiltonian:

$$
  H = -J\sum_{\braket{ij}}(S_i^xS_i^x + S_i^yS_j^y)
$$

where $\braket{ij}$ denotes nearest neighbor pairs over a one dimensional lattice of spins. The main innovation of the paper is the incorporation of the Hamiltonian's $U(1)$ symmetry within the network's architecture, which greatly improves its convergence.

# How to run

To begin training, enter at the command line:

```
Python main.py -json <param.json file> -system_size <N> -results_path <res_directory> -symmetric <Symmetric>
```

Command line parameters:

- `<param.json file>` : List of hyperarameters
- `<N>` : Size of system, Number of spins
- `<res_directory>` : Where to save figures 
- `<Symmetric>` : 1 for symmetry imposed, 0 for no symmetry imposed

The JSON file contains hyperparameters that can be found and changed:

- Learning rate
- Number of epochs
- Display Epochs (When verbose = True, will show progress for every integer * display epochs, I.e if Display Epochs = 10, will display metrics every 10 epochs)
- Model parameters
- Number of hidden units (Integer)
- Verbosity level (1 or 0 [True or false])

# File Overview

`utilities.py` contains utility functions, i.e:
  - Support functions for visualizing results
  - Percentage of samples/data that obeys U(1)-symmetry
  - Computing fidelity of RNN’s wave function
  - Loading needed data/constants
  - Loading observables data such as ground state wave function, and energy difference(for testing fidelity/energy diff)
  - Loading data generated by DMRG(for training/sampling)


`visualizations.py` plots observables/metrics over epochs for different system sizes both for normal RNN and U(1)-symmetry:
  - Loss plot
  - Energy difference plot
  - Number of generated configurations with non-zero magnetizations plot
  - Quantum state fidelity plot

`training.py` provides functions for training the recurrent neural network. Uses stochastic gradient descent to backpropagate through the hidden layers to minimize negative loss likelihood

`main.py` is the main entrypoint to the code:
  - Loads model hyper parameters and data
  - Creates directory to store results
  - Initializes RNN
  - Begins training
  - Displays execution time
  
`RNN.py` contains the recurrent neural network architecture, which is as the class `RNN`

# Folders
- `data` contains data from DMRG 
- `final_results` contains final results presented after training
- `notebooks` contains notebooks that were used to plot results
- `params` contains the json file with the values used to analyze the data

# Contributors
- Jefferson Pule Mendez
- Luc Andre Ouellet
- Uzair Lakhani
- Sam Yu
