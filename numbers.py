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


class numberIDBot:
    entity = None
    inputNames = []
    outputNames = []
    
    interactions = []
    adjustments = []
    neurons = []
    
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
        
        setupNeurons()
        
        
    def setupNeurons(self):
        """ Sets up the default neuron structure according to the inputs, 
            outputs, and weights """
         neurons = []
         
         neurons.append(np.zeros(len(inputNames)))
         
         for weight in weightsPerLayer:
             layer = np.zeros(weight)
             neurons.append(layer)
         
         neurons.append(np.zeeros(len(outputNames)))
         
         
        
    def initAdjustments(self):
        """ Initializes a series of blank matrices the same shape as the 
            interactions matrices, to be used for back propagation """
        for inter in interactions:
            adjustments.append(np.zeros(inter.shape))
    
    
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
            
    def receive(self, inputs, label):
        """ Accepts and stores inputs. Starts pipeline that makes next decision
            *** MAY NEED TO DO SOME SORT OF SANITIZATION OR PREPROCESSING """        
        self.inputs = inputs

        target = np.zeros(10)
        target[label] = 1        
        
        decide(sigmoidize(doInteractions()))
        ### ************* SHOULD DO SOMETHING DIFFERENT HERE
    
    
    def delSigmoid(self, x):
        """ returns derivative of sigmoid function """
        return math.exp(x)/((1+math.exp(x))**2)

    def backProp(self, inputs, outputs, target):
        """ performs back propagation using gradient descent
            neurons is an array s.t. neurons[0] is the input,
            neurons[-1] is the output, and neurons 0 < i < len
            is a *hidden layer* value. There exists at least one
            hidden layer. The first level of back propagation is 
            unique from the others, in that there is no summation"""
        
        
        
        for j in range(0, len(neurons[-1])):
            
            zj = neurons[-1][j]
            
            for k in range(0, len(neurons[-2])):
            
                dCdwjk = neurons[-2][k] * delSigmoid(z) * 2 * (outputs[j] - target[j])
                
        
        for i in reversed(range(0, numHiddenLayers - 1)):
            for j in range(0, weightsPerLayer[i]):
                
        
        
    def cost(self, outputs, target):
        """ compute cost function, in this case, sum of square residuals """
        return np.sum((targets-outputs)**2)
        

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
            *** COME UP WITH A BETTER NAME FOR ME 
            *** MAY NEED TO REVERSE THIS """
        temp = inputs
        for i in range(0, len(interactions)):
            temp = interactions[i].dot(temp)
            neurons[i] = temp
            
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
    