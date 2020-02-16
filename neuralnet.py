# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 02:01:19 2020

@author: James
"""

from typing import Iterable, Optional
import numpy as np

from .layer import Layer

class NeuralNet():
    
    def __init__(self, num_neurons: Iterable[int]):
        """ """
        assert len(num_neurons) >= 2
        assert all([type(x) is int and x > 0 for x in num_neurons])
        
        # create a list
        self.layers = []
        for num_in, num_out in zip(num_neurons[:-1], num_neurons[1:]):
            self.layers.append(Layer(num_in, num_out))