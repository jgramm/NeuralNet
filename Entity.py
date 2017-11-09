# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 08:10:59 2017

@author: James
"""

import numpy as np, random

class entity:
    
    inputNames = []
    outputNames = []
        
    numHiddenLayers = 0
    weightsPerLayer = []
    numBotsPerGen = 0
    numGens = 0
    spareShift = 1  ### ADD ME TO INIT AND GIVE ME A BETTER NAME
                    ### Number of shitty bots to save/good to kill
    
    bots = []
    scores = []
    
    ################## INITIALIZE/BUILD THE BOTS #####################
    
    def __init__(self, inputNames, outputNames, numHiddenLayers, weightsPerLayer, numBotsPerGen, numGens):
        """ FINISH ME """

        ## Must have an even number of bots to make my life easier
        if ((numBotsPerGen % 2) == 1):
            numBotsPerGen += 1
        
        ## Must be enough to kill some off and reproduce
        if (numBotsPerGen < 4):
            numBotsPerGen = 4
            
            
        self.inputNames = inputNames
        self.outputNames = outputNames
        self.numHiddenLayers = numHiddenLayers
        self.weightsPerLayer = weightsPerLayer
        self.numBotsPerGen = numBotsPerGen
        self.numGens = numGens
        self.bots = buildInitBots()
        

        
    def buildInitBots(self):
        """ builds random initial bots """
        temp = []
        for i in range(0, numBotsPerGen):
            temp.append(bot(self, inputNames, outputNames, weightsPerLayer))
        return temp
        
    ################### RUN THE BOT ########################  
    ### THIS SECTION HAS INFORMATION THAT IS SPECIFIC TO ###
    ###  THE TASK BEING PERFORMED. THIS IS WHAT WE WILL  ###
    ### CHANGE ACCORDING TO THE TASK THE NETWORK PERFORMS###
    ########################################################
    

    def run(bot):
        """ Starts a loop that runs the receive command repeatedly, 
            corresponding to the task running. should check for 
            proper termination, and on completion, assigns a fitness 
            score to each of the bots"""
        
    def receive(self, bot, outputs):
        """ accepts output from bot. should check if inputs are valid or not,
            apply inputs, process, send bot new information 
            *** FINISH ME I AM IMPORTANT"""
        return 0
    
    def process(self, bot):
        """ """ 
    
    def stop(self, bot):
        """ """ 
        
    
    ############ LET'S GET DARWINIAN ON THESE BOYS ############
    
    def sortBots(self):
        inds = np.argsort(np.array(scores))
        self.bots = bots[inds]
        self.scores = scores[inds]
        
        
    def killBots(self):
        """ kills most of the bad bots and few of the good according to 
            spareShift variable. returns half the number of bots 
            IF I NEED SCORES IN THE FUTURE, ZIP THEM FIRST AND THEN
            DO THIS """        
        
        half = int(len(bots)/2)
        bad = zip(bots[0:half], scores[0:half])
        good = zip(bots[half:], scores[half:])
        random.shuffle(bad)
        random.shuffle(good)
        bad = bad[0:spareShift]
        good = good[half+spareShift:]
        
        bad.extend(good)
        
        random.shuffle(bad)
        
        return bad
        
    def NAMEME(self, zipped):
        """ WHAT DO I DO? THE WORLD MAY NEVER KNOW. FIX THAT JAMES! """        
        newGen = []
        
        for i in range(0, len(zipped)):
            tempBots = zipped
            b = tempBots.pop(i)
            
            random.shuffle(tempBots)

            ## Procreate once            
            b2 = tempBots.pop(0)
            ratio = b[1] / (b[1] + b2[1])
            kid1 = b.procreate(b, ratio, mutationRate)
            
            ## procreate again
            b2 = tempBots.pop(0)
            ratio = b[1] / (b[1] + b2[1])            
            kid2 = b.procreate(b, ratio, mutationRate)
        
            newGen.append(kid1)
            newGen.append(kid2)
    
    
    
    