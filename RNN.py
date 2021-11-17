"""
Recurrent neural network

Authors: Uzair Lakhani, Luc Andre Ouellet, Jefferson Pule Mendez
"""

import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import json

from data import data_load


class ConventionalRNN(nn.Module):

    def __init__(self,hidden_units, system_size,n_samples, random_seed):
        super(ConventionalRNN, self).__init__()

        self.hidden_units = hidden_units
        self.random_seed  = random_seed
        self.system_size  = system_size
        self.n_samples    = n_samples
        self.global_n_sam = n_samples


        # hidden vector
        

        self.gru_cell = nn.GRUCell(input_size=2, hidden_size=self.hidden_units, bias=True)  # n_h, 2 -> n_h
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=2, bias=True)  # n_h -> 2
        self.softmax = nn.Softmax(dim=1)  # 2 -> 2


    def forward(self, spin, hidden_vec):
        
        """
        Returns a new value for the hidden vector and a set of probabilities 
        
        """
        
        new_hidden_vec = self.gru_cell(spin, hidden_vec)
        linear_output = self.linear(new_hidden_vec)
        probabilities = self.softmax(linear_output) # Probability of getting spin sigma_n+1

        return new_hidden_vec, probabilities

    def sample_or_train(self, RNN_iteration,probabilities, data, training):
        
        """
        Function returns the new spin configuration (sigma_(n+1)), the one hot encoded version of it and the 
        probability of getting that specific output, given probabilities. 
        
        RNN_iteration    =  int  
                            number of spin that the RNN is iterating at the time

        data              =  pytorch tensor
                             The data set for training. (Useful when data is not required). 

        training         =  boolean. 
                            Whether we are training using data or not. If false then just sample randomly.

        sample_sigma_n   =  Tensor of size (ns,1)
                            New set of samples 

        sigma_nhotencoded=  Tensor of size (ns, 2)
                            One hot encoded of sample_sigma_n

        chosen_prob      =  Tensor of size (ns, 2)
                            Probability of obtaining sample_sigma_n on step RNN_iteration
        """
        n=RNN_iteration
 
        if training == True:
            
            sample_sigma_n=data[:,n].long()
            sigma_n_hotencoded  = torch.nn.functional.one_hot(sample_sigma_n, num_classes=2)
            ynsigman = probabilities*sigma_n_hotencoded
            chosen_prob =torch.sum(ynsigman, dim=1, keepdim=True)

        else:
            prob_of_getting_1=probabilities[:,1]

            sample_sigma_n=torch.bernoulli(prob_of_getting_1).long() #Check dimensions            
            sigma_n_hotencoded=torch.nn.functional.one_hot(sample_sigma_n, num_classes=2)
            ynsigmnan=probabilities*sigma_n_hotencoded
            chosen_prob= torch.sum(ynsigmnan, dim=1, keepdim=True)

        sample_sigma_n=torch.reshape(sample_sigma_n,shape=(sample_sigma_n.shape[0],1))
        
        return sample_sigma_n.float(), sigma_n_hotencoded.float(),chosen_prob.float()   

    def N_samples(self,nsamples, training, data=1):

        """
        Function sequentially performs system_size iterations returning for each iteration a sample sigma_i and its conditional probablity. 
        The output are two tensors containing the probabilities of a full configuration sigma and the individual
        conditional probabilities. 
        
        nsamples          =  int  
                             number of samples that we want as an output 

        training          =  boolean. 
                             Whether we are training using data or not. If false then just sample randomly.

        data              =  pytorch tensor
                             The data set for training. It will be equal to 1 if not data is provided (Useful when data is not required). 

        Probabilities     =  Tensor of size (nsamples,size_of_system)
                             Probability of each configuration sigma 

        Sampled_Spins     =  Tensor of size (nsamples,size_of_system)
                             Configuration sigma 

        """
        print("starting ns=",self.n_samples)
        if training==True:
            self.n_samples=data.shape[0]
            print(self.n_samples)
        print("working with ns=",self.n_samples)

        Probabilities = torch.zeros(size=(self.n_samples,1))
        Sampled_Spins = torch.zeros(size=(self.n_samples,1))

        for n in range(self.system_size):

            # Initialize the hidden variable and the initial spin for RNN

            input_batch = torch.zeros(size=(self.n_samples,), ).long()
            self.h_n = torch.zeros(size=(self.n_samples, self.hidden_units)).float()
            self.sigma_n = torch.nn.functional.one_hot(input_batch, num_classes=2).float()


            new_hidden_vec, prob = self.forward(self.sigma_n,self.h_n)
            sampled_spin, spin_hoten, chosen_prob= self.sample_or_train(n, prob, data, training)
            
            if n==0:

                Probabilities=torch.add(Probabilities, chosen_prob)
                Sampled_Spins=torch.add(Sampled_Spins, sampled_spin)

            else: 
                Probabilities=torch.cat([Probabilities, chosen_prob], dim=1)
                Sampled_Spins=torch.cat([Sampled_Spins, sampled_spin], dim=1)
            
            self.h_n     = new_hidden_vec
            self.sigma_n = spin_hoten

        #reset self.n_samples
        self.n_samples=self.global_n_sam

        print(training,"Prob",Probabilities)
        print(training,"Samp", Sampled_Spins)
        print(training,"data", data)
            
        return Sampled_Spins, Probabilities 


    def get_data(self):
        return

    def backprop(self):
        return


if __name__ == "__main__":

    hidden_units = 100
    random_seed = 1
    system_size = 4
    n_samples = 5

    model = ConventionalRNN(hidden_units, system_size,n_samples, random_seed)
    data=torch.tensor([[1,0,0,1],[0,1,1,0],[1,0,0,1],[0,1,1,0],[1,1,1,1],[1,1,1,1]]) #DATA SHOULD BE OUR BATCHES IN TRAINING
    P,S=model.N_samples(model.n_samples,True,data)  #INCLUDE DATA WHEN TRAINING TRUE
    P,S=model.N_samples(model.n_samples,False)      #DO NOT INCLUDE DATA WHEN TRAINING NOT TRUE