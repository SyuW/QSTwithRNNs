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

def U_1_RNN_fidelity(N):
    
    with open(f"results/N={N}/N={N}psi_N={N}_RNN.pkl", "rb") as f:
        psi_RNN=pickle.load(f)

    with open(f"results/N={N}/N={N}psi_N={N}_U(1).pkl", "rb") as file:
        psi_U_1=pickle.load(file)
    print(len(psi_RNN))
    print(len(psi_U_1))
    fidelity=[]
    
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

                fidelity_epoch += psi_U_1_dic[i]*psi_RNN_dic[i]

        fidelity.append(fidelity_epoch**2)
    
    fidelity=np.array(fidelity)
    np.save(f"results/N={N}/infidelity_N_{N}_U_1_RNN.npy",1-fidelity)

    return fidelity

if __name__ == "__main__":
    pass
