# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 06:19:26 2017

@author: James
"""

import random, numpy as np


""" Network has an input layer, hidden layers, and an output layer
    
    Weights between layers are labeled interactions and are represented
    as matrices for computational ease.
    
    
"""


class bot:
    entity = None
    inputNames = []
    outputNames = []
    
    interactions = []
    
    numHiddenLayers = 0
    weightsPerLayer = []
    
    inputs = []
    ################## Initialization ################
    
    def __init__(self, entity, inputNames, outputNames, weightsPerLayer=[10]):
        """ entity is parent object that bot interacts with (thing we are
            trying to solve with this bot)
            weightsPerLayer is an  array of ints corresponding to 
            number of weights in each layer"""
        self.entity = entity
        self.inputNames = inputNames
        self.outputNames = outputNames
        self.numHiddenLayers = len(weightsPerLayer)
        self.weightsPerLayer = weightsPerLayer
        
    
    def giveInteractions(self, interactions):
        self.interactions = interactions
    
    def generateInteractions(self):
        """ generates a random set of weights for our network to begin
            working with. numLayers is an int and 
            *** ADJUST THIS TAG POST REFACTORING
        """
        tempInteractions = []
    
        for i in range(0, numHiddenLayers+1):
            j = weightsPerLayer[i+1]
            k = weightsPerLayer[i]
            temp = np.zeros((j, k)) 
            
            for m in range(0, j):
                for n in range(0, k):
                    temp[m,n] = 2 * random.random() - 1
                    
            tempInteractions.append(temp)
            
        self.interactions = tempInteractions
        



    ############## ON FRAME INTERACTIONS ###################
            
    def receive(self, inputs):
        """ Accepts and stores inputs. Starts pipeline that makes next decision
            *** MAY NEED TO DO SOME SORT OF SANITIZATION OR PREPROCESSING """        
        self.inputs = inputs
        
        decide(sigmoidize(doInteractions()))


    def decide(self, sigmoids):
        """ takes sigmoidized data and sends a decision (array of 1's, 0's)
            about what outputs to select 
            *** FINISH ME, currently applies a fixed bias of .5 after 
            sigmoidization. def want biases before and not fixed. may
            also want other behaviors here """
        entity.receive(sigmoids > .5)
    
    
    def doInteractions(self):
        """ performs matrix multiplication of each interaction between
            input and output layers 
            *** COME UP WITH A BETTER NAME FOR ME """
        temp = inputs
        for i in range(0, len(interactions)):
            temp = interactions[i].dot(temp)
    
        return output
    
    
    def sigmoidize(self, outputs):
        """ takes a list of values and applies sigmoid function """
        for i in range(0, len(outputs)):
            outputs[i] = 1/(1+math.exp(-outputs[i]))
        
        return outputs


###################### OTHER #############################
    
    
    def procreate(self, other, prefRatio, mutationRate):
        """ takes another bot and makes a  """
        raise Exception("FILL ME OUT JAMES")
    