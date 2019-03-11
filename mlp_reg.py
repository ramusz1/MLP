# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:26:10 2019

@author: Piotr
"""
from mlp import MLP
from visuals.plotter import LossPlotter
#from visuals.graph import NetworkGraph
import numpy as np
import sys
import matplotlib.pyplot as plt

class MLPRegressor(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def train(self, x, y, plotLoss = False):   
        if(x.ndim == 1):
            x = x[:, np.newaxis]
            y = y[:, np.newaxis]
#        x,mu,sigma = self.normalize(x)
        for i in range(self.maxIter):
            loss = self.trainEpoch(x,y)
            sys.stdout.write('%d '% loss)
            sys.stdout.flush()
        print('')
        
    def forward(self, x):
        aList = []
        zList = []
        for i in range(self.layersCount-1):
            aList.append(x)
            z = np.matmul(x, self.weights[i])
            if self.usesBias:
                z = z + self.bias[i]
            x = self.activation(z)
            zList.append(z)
        #w ostatniej warstwie nie aktywujemy 
        aList.append(zList[-1])
        return aList, zList
    
    def normalize(self,x):
        l = len(x);
        mu = np.mean(x);
        sigma = np.std(x);
        return (x-mu)/sigma, mu, sigma

    def backprop(self, a, z, y):
        derivativeChain = self.getLossDerivative(a[-1], y)
        deltas = []
        biasDeltas = []
        for l in range(1, self.layersCount):
            # activation function backprop
            if(l != 1):
                derivativeChain = derivativeChain * self.activationDx(z[-l])
            else: #dla ostatniego pochodna jest 1, bo nie ma funkcji aktywacji 
                derivativeChain = derivativeChain 
            
            # bias gradient is ready
            biasDeltas.append(np.sum(derivativeChain, axis = 0))
            # matrix multiplication X*W backprop
            # gradient for W
            delta = np.matmul(a[-l-1].T, derivativeChain)
            # debug print('W shape' ,self.weights[-l].shape, 'vs', delta.shape)
            deltas.append(delta)
            # gradient for X
            if l < self.layersCount - 1:
                derivativeChain = np.matmul(derivativeChain, self.weights[-l].T)
    
        return deltas, biasDeltas

    
mlp = MLPRegressor([1, 4, 1], usesBias = True)
mlp.train(training_x, training_y, False)
plt.plot(test_x,mlp.predict(test_x[:,np.newaxis]),'.')
plt.figure(0)
plt.plot(test_x, test_y)
