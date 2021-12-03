"""
Functions for visualizing the results

Authors: Kishor Srikantharuban, Sam Yu
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def create_energy_plot(num_spins, in_path, fig_no):

    path = os.path.join(in_path, f"N={num_spins}")

    # load the energy values from array
    energy_diff_symm_false = np.load(os.path.join(path, f"energy_diff_{num_spins}_symm_False.npy"))
    energy_diff_symm_true = np.load(os.path.join(path, f"energy_diff_{num_spins}_symm_True.npy"))
    epochs = np.arange(len(energy_diff_symm_false)) + 1

    with plt.ioff():
        fig, ax = plt.subplots()

    # plot everything
    nth = 19

    # plot using the style of figure 4 from paper
    if fig_no == 4:
        ax.semilogy(epochs[::nth], energy_diff_symm_false[::nth], marker=">", label="RNN")
        ax.semilogy(epochs[::nth], energy_diff_symm_true[::nth], marker="<", label="U(1)-RNN")

    # plot using the style of figure 5 from paper
    elif fig_no == 5:
        ax.plot(epochs[::nth], energy_diff_symm_false[::nth], marker=">", label="RNN")
        ax.plot(epochs[::nth], energy_diff_symm_true[::nth], marker="<", label="U(1)-RNN")
        ax.set_ylim([0.000, 0.06])
    else:
        pass

    ax.set_xlabel("Epochs")
    ax.set_ylabel(r"Energy difference, $\epsilon$")
    ax.set_title(f"Energy difference during training (N = {num_spins})")
    ax.legend()

    return fig


def plot_loss_values(num_epochs, vals, N):
    """
    Plot the loss values from training

    :param num_epochs:
    :param vals:
    :param N:
    :return:
    """
    with plt.ioff():
        fig, ax = plt.subplots()

    ax.plot(np.arange(num_epochs), vals, marker=">", color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log Loss")
    ax.set_title(f"Loss vs epoch for N={N}")

    return fig


def plot_nonzero_sz(num_epochs, vals, N):
    """
    Plot the percentage of samples with non-zero magnetization

    :param num_epochs:
    :param vals:
    :param N:
    :return:
    """
    with plt.ioff():
        fig, ax = plt.subplots()

    ax.plot(np.arange(num_epochs), vals, color="blue", merker='<')
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Percentage of samples with $S_z \neq 0$")
    ax.set_title(r"Fraction of samples with $S_z \neq 0$ " + f"for N={N}")

    return fig


if __name__ == "__main__":

    results_path = "final_results/"

    # figure 4 from paper
    for N in [4, 10]:
        energy_fig = create_energy_plot(num_spins=N, in_path=results_path, fig_no=4)
        energy_fig.savefig(os.path.join("figures", f"energy_diff_N_{N}.png"))

    # figure 5 from paper
    for N in [20, 30]:
        energy_fig = create_energy_plot(num_spins=N, in_path=results_path, fig_no=5)
        energy_fig.savefig(os.path.join("figures", f"energy_diff_N_{N}.png"))

