# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:12:55 2019

@author: Piotr
"""
import numpy as np
import functions as fn
from mlp import MLP
from layer import *
import csv
from os import path

log = 'gridSearchLog.csv'

## train_y_acc and train_y can be dfferent - one hot encoding in train_y, but not in the second one
## *_acc is for calculating accuracy
def gridSearch(layersSet, alpha_vec, batch_size_vec, eta_vec,
      train_x, train_y, test_x, test_y, train_y_acc, test_y_acc,
      lossFunction):
    error = []
    for layers in layersSet:
        for alpha in alpha_vec:
            for batch in batch_size_vec:
                for eta in eta_vec:
                    for layer in layers:
                        layer.reset()
                    mlp = MLP(
                              layers=layers,
                              lossFunction=lossFunction,
                              learningRate=alpha,
                              lrDecay=0.98,
                              eta=eta,
                              batchSize=batch,
                              maxIter=500
                         )
                    mlp.train(train_x, train_y, test_x, test_y, plotLoss=False)
                    acc_tr = mlp.accuracy(train_x, train_y_acc)
                    acc_test = mlp.accuracy(test_x, test_y_acc) 
                    modelName = "layers"+ '_'.join([str(i) for i in layers]) + \
                        "alpha"+ str(alpha) + \
                        "batch"+ str(batch) + \
                        "eta"  + str(eta)
                    error = [[modelName, acc_tr, acc_test]]
                    with open(log, 'a+', newline='') as file:
                        print(error)
                        wr = csv.writer(file)
                        wr.writerows(error)

def getLayers(inputSize, hiddenWidth, outputSize):
    return [
        FullyConnected(inputSize, hiddenWidth),
        Activation(fn.sigmoid()),
        FullyConnected(hiddenWidth, outputSize),
    ]

def getMoreLayers(inputSize, hiddenWidth, outputSize):
    return [
        FullyConnected(inputSize, hiddenWidth),
        Activation(fn.relu()),
        FullyConnected(hiddenWidth, hiddenWidth),
        Activation(fn.relu()),
        FullyConnected(hiddenWidth, outputSize),
    ]

def runGridSearch(train_x, train_y_one_hot, train_y, test_x,  test_y_one_hot, test_y, input_size, output_size, lossFunction):
    layers = list(map( lambda hidden: getLayers(input_size,hidden,output_size), [4, 8, 12, 16, 20, 30]))
    moreLayers = list(map( lambda hidden: getMoreLayers(input_size,hidden,output_size), [8, 16]))
    layers = layers + moreLayers
    gridSearch( 
        layersSet = layers,
        alpha_vec = [0.01, 0.05, 0.1],
        batch_size_vec = [6, 12, 36],
        eta_vec = [0, 0.1, 0.2],
        train_x = train_x,
        train_y = train_y_one_hot,
        test_x = test_x,
        test_y = train_y_one_hot,
        train_y_acc = train_y,
        test_y_acc = test_y,
        lossFunction = lossFunction,
    )
