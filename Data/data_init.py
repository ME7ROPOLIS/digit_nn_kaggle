# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 20:54:38 2016

@author: Jake Fortner
"""

import pandas as pd
import numpy as np
import os

def get(hidden_layers, output_layer):
    
    # Load Training & Testing Sets #
    train_dir = os.getcwd() + '\\Data\\train.csv'    
    test_dir = os.getcwd() + '\\Data\\test.csv'    
    
    train = pd.read_csv(train_dir)    
    
    test = pd.read_csv(test_dir)
    
    # Set X and y #
    X = np.matrix(train.ix[:40000, 'pixel0':])
    y = np.matrix(train.ix[:40000, 'label'])
    X_cv = np.matrix(train.ix[40000:, 'pixel0':])
    y_cv = np.matrix(train.ix[40000:, 'label'])


    # Set Layer Values #
    input_layer = np.size(X, 1)
    hidden_nodes = hidden_layers + 23 # Num. Randomly Selected; could change
    
    # Create  Weights Info #    
    layers = []
    layers.append(input_layer)
    
    for i in range(hidden_layers):
        layers.append(hidden_nodes)
        
    layers.append(output_layer) # At a later date, make this layer pythonic
    
    weights_info = [(layers[i + 1], layers[i]) for i in range(len(layers) - 1)]
    weights = [(layers[i + 1], layers[i] + 1) for i in range(len(layers) - 1)]
    
    # Create Thetas #
    orig_thetas = []
    for i in range(len(weights_info)):
        orig_thetas.append(np.matrix(np.random.randn(*weights_info[i])))
        orig_thetas[i] = orig_thetas[i] / np.sqrt(X.shape[1])
        orig_thetas[i] = np.insert(orig_thetas[i], 0, np.random.randn(), 1)
 
    return X, y, X_cv, y_cv, orig_thetas, weights, test
    
    
    
    
    