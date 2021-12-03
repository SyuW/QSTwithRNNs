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


def plot_loss_values(num_spins, in_path):
    """
    Plot the loss values from training

    :param num_spins:
    :param in_path:
    :return:
    """
    with plt.ioff():
        fig, ax = plt.subplots()

    path = os.path.join(in_path, f"N={num_spins}")
    loss_symm_true = np.load(os.path.join(path, f"loss_N_{num_spins}_symm_True.npy"))
    loss_symm_false = np.load(os.path.join(path, f"loss_N_{num_spins}_symm_False.npy"))

    # plot everything
    epochs = np.arange(len(loss_symm_true)) + 1
    ax.plot(epochs, loss_symm_true, marker=">", color="red", markevery=100, label="U(1)-RNN")
    ax.plot(epochs, loss_symm_false, marker="<", color="blue", markevery=100, label="RNN")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log Loss")
    ax.set_title(f"Loss vs epoch for N={N}")

    return fig


def plot_nonzero_sz(num_spins, in_path):
    """
    Plot the percentage of samples with non-zero magnetization

    :param num_spins:
    :param in_path:
    :return:
    """

    with plt.ioff():
        fig, ax = plt.subplots()

    path = os.path.join(in_path, f"N={num_spins}")
    sz_symm_false = np.load(os.path.join(path, f"sz_N_{num_spins}_symm_False.npy"))

    epochs = np.arange(len(sz_symm_false)) + 1
    ax.plot(epochs[::20], sz_symm_false[::20], color="blue", marker='<', markevery=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Percentage of samples with $S_z \neq 0$")
    ax.set_title(r"Fraction of samples with $S_z \neq 0$ " + f"for N={N}")

    return fig


if __name__ == "__main__":

    results_path = "final_results/"

    # figure 4 from paper - energy differences
    for N in [4, 10]:
        energy_fig = create_energy_plot(num_spins=N, in_path=results_path, fig_no=4)
        energy_fig.savefig(os.path.join("figures", f"energy_diff_N_{N}.png"))

    # figure 5 from paper - energy differences
    for N in [20, 30]:
        energy_fig = create_energy_plot(num_spins=N, in_path=results_path, fig_no=5)
        energy_fig.savefig(os.path.join("figures", f"energy_diff_N_{N}.png"))

    # figure 6 from paper - loss plots
    for N in [4, 10]:
        loss_fig = plot_loss_values(num_spins=N, in_path=results_path)
        loss_fig.savefig(os.path.join("figures", f"loss_N_{N}.png"))

    # figure 6 from paper - sz non-zero plots
    for N in [4, 10]:
        sz_nonzero_fig = plot_nonzero_sz(num_spins=N, in_path=results_path)
        sz_nonzero_fig.savefig(os.path.join("figures", f"sz_N_{N}.png"))

