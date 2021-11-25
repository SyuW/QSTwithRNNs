"""
Functions for visualizing the results

Authors: Kishor Srikantharuban, Sam Yu
"""

import matplotlib.pyplot as plt


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

    ax.plot(range(num_epochs), vals, marker=">", color="red")
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

    ax.plot(range(num_epochs), vals, color="blue", merker='<')
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Percentage of samples with $S_z \neq 0$")
    ax.set_title(r"Fraction of samples with $S_z \neq 0$ " + f"for N={N}")

    return fig


if __name__ == "__main__":
    pass
