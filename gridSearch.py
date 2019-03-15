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

def gridSearch(n, layers, alpha_vec, batchSizeVec, eta_vec, tr_x,tr_y,ts_x,ts_y):
    error = []
    for layer in layers:
        for alpha in alpha_vec:
            for batch in batchSizeVec:
                for eta in eta_vec:
                    mlp = MLP(
                              layers=layer,
                              lossFunction=Loss(fn.crossEntropyWithSoftmax()),
                              learningRate=alpha,
                              lrDecay=0.9,
                              eta=0.9,
                              batchSize=batch,
                              maxIter=500
                         )
                    mlp.train(tr_x, tr_y, ts_x, ts_y, plotLoss=True)
                    acc_tr = mlp.accuracy(tr_x, tr_y)
                    acc_ts = mlp.accuracy(ts_x, ts_y) 
                    log = "layers"+ '_'.join([str(i) for i in layer]) + \
                          "alpha"+ str(alpha) + \
                          "batch"+ str(batch) + \
                          "eta"  + str(eta)
                    error.append([acc_tr, acc_ts])
                    
                    try:
                        file = open(log+'.csv', 'a',newline='')
                    except IOError:
                        file = open(log+'.csv', 'w',newline='')
                    wr = csv.writer(file)
                    wr.writerows(error)