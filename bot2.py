# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:53:13 2017

@author: James
"""

import random, numpy as np
from keras.datasets import mnist


""" 
    *** GIVE ME SOME GOOD DOCUMENTATION
"""

class layer:
    
    previousL = None
    nextL = None
    inputs = None
    outputs = None
    weights = None
    biases = None
    wCorrections = None
    bCorrections = None
    bPropCount = 0
    learningRate = 1
    
    ####################### SET UP LAYERS #########################
    
    def __init__(self, prev, inputCount, outputCount):
        """ Initialize empty arrays and link layers """
        self.previousL = prev
        self.inputs = np.zeros(inputCount)
        self.outputs = np.zeros(outputCount)
        
        
        self.weights = np.zeros((outputCount, inputCount))
        self.wCorrections  = np.zeros((outputCount, inputCount))
        
        self.biases = np.zeros(outputCount)
        self.bCorrections = np.zeros(outputCount)
        
        if(not(self.previousL is None)):
            self.previousL.receiveNextLayer(self)
        
    def receiveNextLayer(self, nextL):
        """ Receives and links the next layer. This will be called back from 
            the next layer, and won't be called at all for the last layer """
        self.nextL = nextL
    
    def receiveInputs(self, inputs):
        """ Receives inputs for this layer """
        if(not(inputs.shape == self.inputs.shape)):
            print(inputs.shape, self.inputs.shape)
            raise ValueError("inputs are not the correct size")
            
        self.inputs = inputs
        #print('POOPIES', self.inputs)
    
    
    def initializeRandomWeights(self):
        for j in range(0, len(self.outputs)):
            for k in range(0, len(self.inputs)):
                self.weights[j,k] = 2 * random.random() - 1

    
    ######################### PROPAGATION #############################

    def sigmoid(self, x):
        """ Uses the logistic function to squash values """
        return 1/(1 + np.exp(-x))
    
    def delSigmoid(self, x):
        """ Returns derivative of logistic function, which has property 
            that dSig(x) = sig(x) * (1-sig(x)) """
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def fProp(self):
        """ Performs forward propagation to generate outputs, and sends those 
            to the next layer """
    
        self.outputs = self.sigmoid(self.weights.dot(self.inputs) + self.biases)
        
        ### If the end, return, if not, recurse
        if(self.nextL is None):
            return self.outputs
        else:
            self.nextL.receiveInputs(self.outputs)
            return self.nextL.fProp()
    
    
    def bProp(self, desiredOut):
        """ Performs back propagation for this layer 
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
            
            
    def correct(self):
        """ Apply average corrections that have been accumulating via backprop
            Reset correction terms and count to 0"""
        self.weights += self.learningRate * self.wCorrections / self.bPropCount
        self.biases += self.learningRate * self.bCorrections / self.bPropCount 
        
        self.bPropCount = 0
        self.wCorrections = np.zeros(self.wCorrections.shape)
        self.bCorrections = np.zeros(self.bCorrections.shape)
    

class Bot:
    """ A bot is a list of layers 
        A layer has an input, an interaction matrix, and an output
        Each layer is linked to the next"""
    
    layers = [] 
    
    def __init__(self, nodesPerLayer):
        """ nodesPerLayer is a list of ints s.t. the first number is the number
            of inputs, the last is the number of outputs, and intermediate 
            numbers correspond to nodes in the hidden layers """
        
        self.layers = []
        self.setupLayers(nodesPerLayer)    
        
    
    
    def setupLayers(self, nodesPerLayer):
        """ Initializes and links all layers together. Also generates random
            weight matrices at each layer """
        
        for i in range(0, len(nodesPerLayer) - 1):
        #    print('i', i)
            prev = None
            if(i > 0):
        #        print('setup layers', i)
                prev = self.layers[i-1]
                prev.receiveNextLayer(self)
            
            L = layer(prev, nodesPerLayer[i], nodesPerLayer[i+1])
            
            L.initializeRandomWeights()
            
            self.layers.append(L)
        #    print('len layers', len(self.layers))
            
    
    def runOnce(self, inputs, desiredOut):
        
        self.layers[0].receiveInputs(inputs)
        #print(self.layers[0].inputs)
        
        result = self.layers[0].fProp()
        
        #print(result, desiredOut)
        
        self.layers[-1].bProp(desiredOut)
        
        return result
    
    def correct(self):
        for l in self.layers:
            l.correct()
        
            


def run(b):
    data = mnist.load_data()
    d0 = data[0]
    
    
    
    for i in range(0, len(d0[0])):
        desired = np.zeros(10)
        desired[d0[1][i]] = 1
    
        l = d0[0][i]
        inputs = np.asarray([item for sublist in l for item in sublist])
        inputs = inputs / 256.
        b.runOnce(inputs, desired)    
        
        if(b.layers[0].bPropCount % 10 == 9):
            b.correct()
            
        
        if(i % 500 == 0):
            print(i)
        
    test()
        
def test(b):
    data = mnist.load_data()
    for i in range(0, 20):
        l = data[1][0][i]
        desired = np.zeros(10)
        desired[data[1][1][i]] = 1
        
        inputs = np.asarray([item for sublist in l for item in sublist])
        inputs = inputs / 256.
        out = b.runOnce(inputs, desired) 
        
        print(i)
        print(out)
        print(desired)
        

if __name__=='__main__':
    #run()
    print(1)
