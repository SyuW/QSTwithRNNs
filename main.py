"""
Main entry point

Authors: Sam Yu
"""

import argparse
import os
import json

# custom imports
from RNN import RNN
from training import train
from utilities import load_data, load_observables

if __name__ == "__main__":

    # command-line argument parsing
    parser = argparse.ArgumentParser(description="Train RNN for Quantum State Tomography")
    parser.add_argument("-json", default="params/params.json", help="input path to json file")
    parser.add_argument("-system_size", type=int, default=4, help="Size of our system. Default 10")
    parser.add_argument("-results_path", default="results", help="file path to results")
    parser.add_argument("-symmetric", type=int, help="Whether to use RNN or U(1)-RNN. 0 - RNN, 1 - U(1)")
    args = parser.parse_args()

    # input screening
    if args.system_size not in [2, 4, 6, 8, 10, 16, 20, 30, 40, 50]:
        raise ValueError(f"Chosen system size {args.system_size} not in data")

    # Load the model parameters
    with open(args.json, 'r') as f:
        params = json.load(f)

        lr = params['training']['learning rate']
        epochs = params['training']['epochs']
        de = params['training']['display epochs']
        hidden_units = params['model']['hidden units']
        batch_size = params['data']['batch size']

    # make the directory to store results at
    save_path = os.path.join(args.results_path, f"N={args.system_size}")
    os.makedirs(save_path, exist_ok=True)

    # create the data loader
    data_loader = load_data(f"data/samples_N={args.system_size}_batch=1", params['data']['batch size'])
    gs_psi, dmrg_energy = load_observables(args.system_size)

    # initialize the model
    rnn = RNN(hidden=hidden_units, system_size=args.system_size, symmetric=args.symmetric)

    # start training
    import time

    start = time.time()
    train(rnn, data=data_loader, results_path=save_path, num_epochs=epochs, truth_energy=dmrg_energy,
          truth_psi=gs_psi, learning_rate=lr, display_epochs=de)

    print(f"Execution time: {time.time() - start} seconds")
