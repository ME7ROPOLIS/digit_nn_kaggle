# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 22:51:05 2016

@author: Jake Fortner
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder as onehot

def cost(all_thetas, weights, X, y, lamb):    
    
    thetas = unpack_thetas(all_thetas, weights)

    X = X/255

    # Create Bias #
    X_bias = np.insert(X, 0, 1, 1)
    
    # Set Binary Vectors #
    encoder = onehot(sparse=False)
    y_nn = encoder.fit_transform(y.T)

    
    # Activation Layers #
    act_layers = activation_layers(X_bias, thetas)    

       
    # Create The Peices of the Function #    
    first = np.multiply(-y_nn, np.log(act_layers[-1]))
   
    second = np.multiply(1 - y_nn, np.log(1 - act_layers[-1]))

    
    # REG most likey not working: Greatly increseas Cost when Lamb set to 1
    reg_1 = lamb / (2 * len(X))
    reg_2 = np.power(thetas[0][...,1:], 2).sum()
    for i in range(1, len(thetas)):
        reg_2 = reg_2 + np.power(thetas[i][...,1:], 2).sum()
        
    J = 1 / len(X) * (first - second).sum() + (reg_1 * reg_2)
    print('Current Cost:')
    print(J)
    print('=' * 20)
  
    return J
    
    
def gradient(all_thetas, weights, X, y, lamb):
    
    thetas = unpack_thetas(all_thetas, weights)
    
    X = X/255
    
    # Create Bias #
    X_bias = np.insert(X, 0, 1, 1)
    
    # Set Binary Vectors #
    encoder = onehot(sparse=False)
    y_nn = encoder.fit_transform(y.T)

    # Activation Layers #
    act_layers = activation_layers(X_bias, thetas)
    
    
    # Create Thetas for Deltas #
    theta_delta = []

    for i in range(len(thetas)):
        theta_delta.append(thetas[i][:, 1:])
        
    # Create Gradients of Thetas #
    d = []
    Delta = []
    theta_grad = []
    
    for i in range(len(theta_delta)):
        
        # Set lower-deltas -----             
        if i == 0:
            d.insert(0, act_layers[-(i+1)] - y_nn)  # Work Backwards through act_layers

        else:
            act_temp = act_layers[-(i+1)][:, 1:]
            
            d.insert(0, np.multiply(d[0] * theta_delta[-i], np.multiply(
                act_temp, (1-act_temp))))
           
           
        # Create Deltas -----
        Delta.insert(0, d[0].T * act_layers[-(i+2)])


        # Create Theta_Grad -----
        theta_grad.insert(0, Delta[0] / len(y_nn))

    # Add Bias to / Update Theta_deltas #
    for i in range(len(theta_grad)):
        theta_delta[i] = (lamb/len(y_nn)) * theta_delta[i]
        theta_grad[i] += np.insert(theta_delta[i], 0, 0, 1)


    # Pack Thetas into a Single Vector #    
    gradient_thetas = pack_thetas(theta_grad)


    return gradient_thetas         


def forward_propagate(all_thetas, weights, X, y):
    
    thetas = unpack_thetas(all_thetas, weights)
    
    X = X/255
    
    # Create Bias #
    X_bias = np.insert(X, 0, 1, 1)

    # Activation Layers #
    act_layers = activation_layers(X_bias, thetas)
            
    predict = np.argmax(act_layers[-1], axis=1)

    print(predict[:10])
    print(y[:10].T)
        
    correct = [1 if a == b else 0 for (a, b) in zip(predict, y.T)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))  
    return 'accuracy = {0}%'.format(accuracy * 100) 

       
def sigmoid(z):
    return 1/(1 + np.exp(-z))
    

def activation_layers(X_bias, thetas):
    act_layers = []
    act_layers.append(X_bias)
    for i in range(len(thetas)):

        z = act_layers[i] * thetas[i].T
       
        act_layers.append(sigmoid(z))
        if i != (len(thetas) - 1):
            act_layers[i+1] = np.insert(act_layers[i + 1], 0, 1, 1)
    return act_layers

    
def pack_thetas(thetas):
    new_thetas = np.matrix(np.ravel(thetas[0])).T
    
    for i in range(1, len(thetas)):
        new_thetas = np.concatenate((new_thetas, np.matrix(np.ravel(thetas[i])).T), axis=0)

    return new_thetas

    
def unpack_thetas(all_new_thetas, weights): 
    theta_temp = []
    temp = 0
    wght_totals = [l * m for l, m in weights]


    for i in range(len(weights)):
        theta_temp.append(np.reshape(all_new_thetas[temp:temp + wght_totals[i]], weights[i]))
        temp += wght_totals[i]
   
    return theta_temp
    
