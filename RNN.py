"""
Recurrent neural network

Authors: Uzair Lakhani, Luc Andre Ouellet, Jefferson Pule Mendez
"""

import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim


class ConventionalRNN(nn.Module):

    def __init__(self):
        super(ConventionalRNN, self).__init__()

        hidden_units = 100
        random_seed = 1
        system_size = 10
        n_samples = 1000
      

        # hidden vector
        

        self.gru_cell = nn.GRUCell(input_size=2, hidden_size=hidden_units, bias=True)  # n_h, 2 -> n_h
        self.linear = nn.Linear(in_features=hidden_units, out_features=2, bias=True)  # n_h -> 2
        self.softmax = nn.Softmax()  # 2 -> 2

        # Initialize the hidden variable and the initial spin for RNN

        input_batch = torch.zeros(size=(n_samples,), ).long()
        self.h_n = torch.zeros(size=(n_samples, hidden_units)).float()
        self.sigma_n = torch.nn.functional.one_hot(input_batch, num_classes=2).float()


    def forward(self, spin, hidden_vec):
        
        """
        Returns a new value for the hidden vector and a set of probabilities 
        
        """
        
        new_hidden_vec = self.gru_cell(spin, hidden_vec)
        linear_output = self.linear(new_hidden_vec)
        probabilities = self.softmax(linear_output) # Probability of getting spin sigma_n+1

        return new_hidden_vec, probabilities

    def sample_or_train(self, RNN_iteration,probabilities, data=1, training):
        
        """
        Function returns the new spin configuration (sigma_(n+1)), the one hot encoded version of it and the 
        probability of getting that specific output, given probabilities. 
        
        RNN_iteration    =  int  
                            number of spin that the RNN is iterating at the time

        data              =  pytorch tensor
                             The data set. If not provided then it takes a value of 1 (Useful when data is not required). 

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
            
            sample_sigma_n=data[:,n]
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
        
        return sample_sigma_n, sigma_n_hotencoded,chosen_prob
   
            
            

            


    def get_data(self):
        return

    def backprop(self):
        return


if __name__ == "__main__":
    model = ConventionalRNN()
