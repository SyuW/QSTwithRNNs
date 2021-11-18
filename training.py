"""
Training procedure

Authors: Sam Yu
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from RNN import ConventionalRNN
from data import data_load

import torch
import torch.nn as nn
import torch.optim as optim
import json


def calculate_loss(input, targets):
    """

    :param input: raw probabilities from RNN model
    :param targets:
    :return:
    """

    # use negative log likelihood for the loss function
    loss_fn = nn.NLLLoss()
    loss_val = loss_fn(input, targets)

    return loss_val


def train(model, dataset, **kwargs):
    """

    :param model:
    :param dataset:
    :param kwargs:
    :return:
    """

    # hyperparameters
    learning_rate = 0.001
    display_epochs = 10
    num_epochs = 100
    batch_size = 50

    # defining the optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # start the training
    for epoch in range(num_epochs):

        # clear gradients
        optimizer.zero_grad()

        predictions = ConventionalRNN()

        # compute the loss
        obj_val = calculate_loss(predictions, targets=2)

        # calculate gradients and update parameters
        loss.backward()
        optimizer.step()

        if (num_epochs + 1) % display_epochs:
            print(f"Epoch [{epoch+1}/{num_epochs}\tLoss: {obj_val:.4f}]")

    with open('results/report.txt', 'w') as report_file:
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train RNN for Quantum State Tomography")
    
    parser.add_argument("-json", default="params/params.json",help="input path to json file")
    parser.add_argument("-size_of_system", default="4" ,help="Size of our system. Default 10")
#    parser.add_argument("-results_path", default="results" ,help="file path to results") We need a results folder :v also maybe a src folder. 
    args = parser.parse_args()
    
    #Load model parameters variables 
    with open(args.json, 'r') as f:
        json_file=json.load(f)
    
        lr          = json_file['optim']['learning rate']   
        random_seed = json_file['optim']['random seed'] #Where do we define the random seed
        epochs      = json_file['optim']['epochs']    

        hidden_units= json_file['model']['hidden units']  
        batch_size  = json_file['data']['batch size']
        system_size = int(args.size_of_system)
        n_samples   = 10                               #Number of samples we want our RNN to use
                                                        #I made our RNN calculate this automatically if training is True that is ns will be compatible with the size 
                                                        #our data set, but if training is false it will give you this number of samples. 

    data=data_load("data/samples_N="+args.size_of_system+"_batch=1", batch_size)

    k=0 #DELETE THIS K AFTER YOU ARE DONE 

    for batch in data:
        model = ConventionalRNN(hidden_units, system_size,n_samples, random_seed)

    ################### RUN ME TO SEE THE MODEL CALCULATE N_samples and N_probabilities only onced ###############################
        if k==0:
            print("k",k)
            Probabilities,Sampled_Spins=model.N_samples(model.n_samples,True,batch)  #INCLUDE DATA WHEN TRAINING TRUE
            Probabilities,Sampled_Spins=model.N_samples(model.n_samples,False)       #DO NOT INCLUDE DATA WHEN TRAINING NOT TRUE
            k=1
            print("ONE BATCH IS FINISHED")
            print("####################################################################################################")
    ################## DELETE ME AFTER YOU UNDERSTAND WHAT IS GOING ON #############################################################


    ################## RUN ME TO SEE HOW IT WORKS WITH A SMALL DATA SET YOU MAY WANT TO COMMENT THE PREVIOUS PART SO YOU ONLY SEE THIS PART ##############    
        #if k==0:
        #    batch=torch.tensor([[1,0,0,1],[0,1,1,0],[1,0,0,1],[0,1,1,0],[1,1,1,1]]) #DATA SHOULD BE OUR BATCHES IN TRAINING
        #    Probabilities,Sampled_Spins=model.N_samples(model.n_samples,True,batch)  #INCLUDE DATA WHEN TRAINING TRUE
        #    Probabilities,Sampled_Spins=model.N_samples(model.n_samples,False)       #DO NOT INCLUDE DATA WHEN TRAINING NOT TRUE
        #    k=1
        #    print("ONE BATCH IS FINISHED")
        #    print("####################################################################################################")
    ################### DELETE ME AFTER YOU ARE DONE #################################################################################################
   
    ################## RUN ME TO SEE THE MODEL CALCULATE N_SAMPLES and N_Probabilities FOR ALL THE BATCHES. I AM THE MOST IMPORTANT ONE DO NOT DELETE ME-USE ME  ##############    

#        Probabilities,Sampled_Spins=model.N_samples(model.n_samples,True,batch)  #INCLUDE DATA WHEN TRAINING TRUE
#        Probabilities,Sampled_Spins=model.N_samples(model.n_samples,False)       #DO NOT INCLUDE DATA WHEN TRAINING NOT TRUE
#        print("ONE BATCH IS FINISHED")
#        print("####################################################################################################")

