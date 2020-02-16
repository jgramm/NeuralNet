# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 01:52:20 2020

@author: James
"""

import numpy as np


def sigmoid(x):
    """ Uses the logistic function to squash values """
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    """ 
    Returns derivative of logistic function, which has property that 
    sigmoid'(x) = sig(x) * (1-sig(x)) 
    """
    return sigmoid(x) * (1 - sigmoid(x))