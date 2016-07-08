# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 20:55:04 2016

@author: Jake Fortner
"""


import numpy as np
from Data import data_init
from scipy.optimize import minimize, check_grad
from functions import cost, gradient, pack_thetas, forward_propagate


class Workstation(object):
    def __init__(self, X=0, y=0, orig_thetas=0, lamb=5, epsilon=0.0001, hidden_layers=2, output_layer=10):
        X, y, X_cv, y_cv, orig_thetas, weights, test = data_init.get(hidden_layers, output_layer)
        self.X = X        
        self.y = y
        self.X_cv = X_cv
        self.y_cv = y_cv
        self.orig_thetas = orig_thetas
        self.test = test
        self.all_thetas = 0
        self.lamb = lamb
        self.epsilon = epsilon
        self.weights = weights
        print('Workstation initialized.')
        
        
    def find_min(self):
        self.all_thetas = pack_thetas(self.orig_thetas)

        print('Minimizing . . .')        
        fmin = minimize(fun=cost, x0=self.all_thetas, args=(self.weights, self.X, self.y, self.lamb), method='TNC', jac=gradient, options={'maxiter':250})
        self.all_thetas = np.matrix(fmin.x)
        self.all_thetas = self.all_thetas.T
        
    def predict_all(self):
        print('Predicting . . .')
        accuracy = forward_propagate(self.all_thetas, self.weights, self.X_cv, self.y_cv)
        print(accuracy)
        
        
if __name__ == '__main__':
    ws = Workstation()
    ws.find_min()
    ws.predict_all()   