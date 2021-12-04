"""
Functions for visualizing the results

Authors: Uzair Lakhani, Sam Yu, Jefferson Pule Mendez
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns

sns.set_theme()


def plot_energy_diffs(num_spins, in_path, scaling, plot_every=1):
    """
    create the plot of energy difference vs epochs

    :param plot_every:
    :param scaling:
    :param num_spins:
    :param in_path:
    :return:
    """

    path = os.path.join(in_path, f"N={num_spins}")

    # load the energy values from array
    energy_diff_symm_false = np.load(os.path.join(path, f"energy_diff_{num_spins}_symm_False.npy"))
    energy_diff_symm_true = np.load(os.path.join(path, f"energy_diff_{num_spins}_symm_True.npy"))
    epochs_symm_false = np.arange(len(energy_diff_symm_false)) + 1
    epochs_symm_true = np.arange(len(energy_diff_symm_true)) + 1

    with plt.ioff():
        fig, ax = plt.subplots()

    # plot using the style of figure 4 from paper
    if scaling == "log":
        ax.semilogy(epochs_symm_false[::plot_every], energy_diff_symm_false[::plot_every], marker=">", label="RNN")
        ax.semilogy(epochs_symm_true[::plot_every], energy_diff_symm_true[::plot_every], marker="<", label="U(1)-RNN")

    # plot using the style of figure 5 from paper
    elif scaling == "normal":
        ax.plot(epochs_symm_false[::plot_every], energy_diff_symm_false[::plot_every], marker=">", label="RNN")
        ax.plot(epochs_symm_true[::plot_every], energy_diff_symm_true[::plot_every], marker="<", label="U(1)-RNN")
        ax.set_ylim([0.000, 0.06])

    else:
        raise ValueError(f"{scaling} is not valid. Choose from [log, normal]")

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
    epochs_symm_true = np.arange(len(loss_symm_true)) + 1
    epochs_symm_false = np.arange(len(loss_symm_false)) + 1

    # plot everything
    ax.plot(epochs_symm_true, loss_symm_true, marker=">", color="red", markevery=100, label="U(1)-RNN")
    ax.plot(epochs_symm_false, loss_symm_false, marker="<", color="blue", markevery=100, label="RNN")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log Loss")
    ax.set_title(f"Loss vs epoch for N={num_spins}")

    return fig


def plot_nonzero_sz(num_spins, in_path, plot_every):
    """
    Plot the percentage of samples with non-zero magnetization

    :param plot_every:
    :param num_spins:
    :param in_path:
    :return:
    """

    with plt.ioff():
        fig, ax = plt.subplots()

    path = os.path.join(in_path, f"N={num_spins}")
    sz_symm_false = np.load(os.path.join(path, f"sz_N_{num_spins}_symm_False.npy"))

    epochs = np.arange(len(sz_symm_false)) + 1
    ax.plot(epochs[::plot_every], sz_symm_false[::plot_every], color="blue", marker='<', markevery=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Percentage of samples with $S_z \neq 0$")
    ax.set_title(r"Fraction of samples with $S_z \neq 0$ " + f"for N={num_spins}")

    return fig


def u_1_rnn_fidelity(num_spins):
    with open(f"results/N={num_spins}/N={num_spins}psi_N={num_spins}_RNN.pkl", "rb") as f:
        psi_RNN = pickle.load(f)

    with open(f"results/N={num_spins}/N={num_spins}psi_N={num_spins}_U(1).pkl", "rb") as file:
        psi_U_1 = pickle.load(file)
    print(len(psi_RNN))
    print(len(psi_U_1))
    fidelity = []

    for epoch in range(len(psi_RNN)):

        fidelity_epoch = 0
        psi_RNN_dic = dict()
        psi_U_1_dic = dict()
        psi_RNN_epoch = np.array(psi_RNN[epoch])
        psi_U_1_epoch = np.array(psi_U_1[epoch])

        for sigma in range(len(psi_RNN_epoch)):
            psi_RNN_dic[int(psi_RNN_epoch[sigma][0])] = psi_RNN_epoch[sigma][1]

        for sigma in range(len(psi_U_1_epoch)):
            psi_U_1_dic[int(psi_U_1_epoch[sigma][0])] = psi_U_1_epoch[sigma][1]

        for i in psi_U_1_dic:

            if i in psi_RNN_dic:
                fidelity_epoch += psi_U_1_dic[i] * psi_RNN_dic[i]

        fidelity.append(fidelity_epoch ** 2)

    fidelity = np.array(fidelity)
    np.save(f"results/N={num_spins}/infidelity_N_{num_spins}_U_1_RNN.npy", 1 - fidelity)

    return fidelity


def plot_infidelity(num_spins, in_path):
    """
    Plot the infidelity during training

    :param num_spins:
    :param in_path:
    :return:
    """

    with plt.ioff():
        fig, ax = plt.subplots()

    path = os.path.join(in_path, f"N={num_spins}")

    symm = ['True', 'False']
    color = ['r', 'b']
    for i in range(len(symm)):
        data = np.load(os.path.join(path, f'infidelity_N_{num_spins}_symm_{symm[i]}.npy'))
        ax.plot(np.arange(2000)[::30], data[::30], marker="v", markevery=1, color=color[i])

    ax.legend(['U(1)-RNN', 'RNN'])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("1-F")
    ax.set_title(f"Infidelity for N={num_spins}")
    ax.set_yscale('log')
    return fig


def plot_infidelity_u1_rnn(num_spins, in_path):
    """
    Plot the infidelity between U1 RNN and RNN during training

    :param num_spins:
    :param in_path:
    :return:
    """

    with plt.ioff():
        fig, ax = plt.subplots()

    path = os.path.join(in_path, f"N={num_spins}")
    data = np.load(os.path.join(path, f'infidelity_N_{num_spins}_U_1_RNN.npy'))
    ax.plot(np.arange(2000)[::30], data[::30], marker="v")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("1-F")
    ax.set_title(f"Plot of infidelity between U1 RNN and RNN during training for N={num_spins}")
    ax.set_yscale('log')
    return fig


if __name__ == "__main__":

    results_path = "final_results/"
    os.makedirs(results_path, exist_ok=True)
    os.makedirs("figures", exist_ok=True)


    # figure 4 from paper - energy differences
    for N in [4, 10]:
        energy_fig = plot_energy_diffs(num_spins=N, in_path=results_path, scaling="log", plot_every=19)
        energy_fig.savefig(os.path.join("figures", f"energy_diff_N_{N}.png"))

    # figure 4 from paper - infidelity plots
    for N in [4, 10]:
        infidelity = plot_infidelity(num_spins=N, in_path=results_path)
        infidelity.savefig(os.path.join("figures", f"infidelity_N_{N}.png"))

    # figure 5 from paper - energy differences
    for N in [20, 30]:
        energy_fig = plot_energy_diffs(num_spins=N, in_path=results_path, scaling="normal", plot_every=19)
        energy_fig.savefig(os.path.join("figures", f"energy_diff_N_{N}.png"))

    # figure 6 from paper - loss plots
    for N in [4, 10]:
        loss_fig = plot_loss_values(num_spins=N, in_path=results_path)
        loss_fig.savefig(os.path.join("figures", f"loss_N_{N}.png"))

    # figure 6 from paper - sz non-zero plots
    for N in [4, 10]:
        sz_nonzero_fig = plot_nonzero_sz(num_spins=N, in_path=results_path, plot_every=20)
        sz_nonzero_fig.savefig(os.path.join("figures", f"sz_N_{N}.png"))

    # figure 4 from paper - infidelity plots
    for N in [4, 10]:
        infidelity = plot_infidelity(num_spins=N, in_path=results_path)
        infidelity.savefig(os.path.join("figures", f"infidelity_N_{N}.png"))

    # infidelity plots between RNN and U1 RNN - requested during midterm presentation
    for N in [4, 10]:
        infidelity_U1_RNN = plot_infidelity_u1_rnn(num_spins=N, in_path=results_path)
        infidelity_U1_RNN.savefig(os.path.join("figures", f"infidelity_U_1_RNN_N_{N}.png"))
