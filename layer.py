# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 01:20:54 2020

@author: James
"""

from typing import Optional
import numpy as np

from .math import sigmoid

class Layer():
    
    def __init__(self, num_in: int, num_out: int, 
                 matrix: Optional[np.ndarray] = None,
                 biases: Optional[np.ndarray] = None):
        
        # Use the given matrix if provided (for testing purposes)
        if matrix is None:
            self.weight_matrix = np.random.randn((num_out, num_in))
        else:
            self.weight_matrix = matrix
        self.num_out, self.num_in = self.matrix.shape
        
        # Use the given biases if provided (for testing purposes)
        if biases is None:
            self.biases = np.zeros((num_out,))
        else:
            assert len(biases) == self.num_out
            self.biases = biases
        
    
    def evaluate(self, input_vec: np.ndarray):
        """ 
        evaluate this layer of the neural network,
        just output = sigmoid(input * weights + biases)
        """
        assert input_vec.shape == (self.num_in,)
        # TODO: linear_result is a crappy name, fix this
        linear_result = np.matmul(self.matrix, input_vec) + self.biases
        return sigmoid(linear_result)
    
    
    
    
    def drop_out(self, drop_out_rate: float):
        """ 
        0 <= drop_out_rate < 1
        during training, randomly set some activations to 0 
        helps network generalize better
        need to figure out at what point to set to zero
        """
        # TODO: Finish me
        assert 0 <= drop_out_rate < 1
        return
        
        
    def back_prop(self, input_vec: np.ndarray, desired_output: np.ndarray):
        """ 
        Performs back propagation for this layer 
        dC/dW is the correction to the weight,
        dC/dB is the correction to the bias,
        dC/dA is the correction to the desired previous input
        k always indexes the inputs
        j always indexes the outputs
    
        C is cost
        Ak is the kth input
        Aj is the jth output = sig(Zj)
        Zj = Sum_k (Wjk*Ak) + Bj
        
        #dC/dAk = sum_j [ dZj/dAk  dAj/dZj  dC/dAj]
        
        #dC/dWjk = [dZj/dWjk  dAj/dZj  dC/dAj]
        #dC/dBj  = [dZj/dBj    dAj/dZj  dC/dAj]
        
        dZ/dB = 1
        dZj/dAk = Wjk 
        dZj/dWjk = Ak
        dAj/dZj = delSig(Zj)
        
        dC/dAj  = Aj - desiredOutj
        """
        
        assert input_vec.shape == (self.num_in,)
        assert desired_output.shape == (self.num_out,)
        # TODO: BELOW CODE IS FROM OLD, TERRIBLE CODE
        # SHOULD BE USED AS A REFERENCE ONLY
        
        ### Weight/bias corrections are based on the outputs moving backwards,  
        ### so we use j indices here
        dCdZjs = np.zeros(len(self.outputs))
        for j in range(0, len(self.outputs)):
            
            Z = self.weights[j].dot(self.inputs) + self.biases[j]
            dCdAj = self.outputs[j] - desiredOut[j]
            dAjdZj = self.delSigmoid(Z)
            
            dCdZjs[j] = dCdAj * dAjdZj              # Store these for next step
            
            dCdWj = dAjdZj * dCdAj * self.inputs    # Weight correction
            dCdBj = dAjdZj * dCdAj                  # Bias correction
            
            self.wCorrections[j] += dCdWj
            self.bCorrections[j] += dCdBj
        
        ### These are corrections to the inputs, and we need to sum over j,
        ### which is why we are using the dot product with the stored 
        ### derivatives from previous step
        dCdAks = np.zeros(len(self.inputs))
        #print(len(self.inputs), self.weights.shape)
        for k in range(0, len(self.inputs)):
            dCdAks[k] = self.weights[:,k].dot(dCdZjs)    # Input correction
        
        ### Keep track of how many corrections this layer has made so we 
        ### can average them later
        self.bPropCount += 1
        
        ### Send previous layer its desired outputs (the back in back prop)
        if(not(self.previousL is None)):
            self.previousL.bProp(self.inputs+dCdAks)