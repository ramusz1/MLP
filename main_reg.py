import numpy as np
from sklearn import datasets
from mlp import MLP
import pandas as pd
from visuals.setVisualization import visualizeSet
from visuals.functionVisualization import visualizeFunction

import functions as fn
from layer import *

def loadDataset(filename):
    dataset = pd.read_csv(filename)
    x = dataset['x'].values
    y = dataset['y'].values
    x = np.expand_dims(x, axis = 1)
    y = np.expand_dims(y, axis = 1)
    return x, y 

training_x, training_y = loadDataset('datasets/regression/data.activation.train.100.csv')
test_x, test_y = loadDataset('datasets/regression/data.activation.test.100.csv')

mlp = MLP(
    layers = [
        FullyConnected(1, 16),
        Activation(fn.sigmoid()),
        FullyConnected(16, 1),
    ],
    lossFunction = Loss(fn.MSE()),
    learningRate = 0.05,
    lrDecay = 1,
    eta = 0.05,
    batchSize = 64,
    maxIter = 200
)

# 2 run options:
# 1. step by step mode with neural network graph
# 2. normal mode:
# - with loss plot
# - with training set visualization
# - only test
import argparse

parser = argparse.ArgumentParser(description='Run mlp classifier training.')

parser.add_argument('--step_by_step', default=False, action='store_true',
                   help='step by step mode with neural network graph visualization')

parser.add_argument('--plot_loss', default=False, action='store_true',
                   help='show live plot of the loss')

parser.add_argument('--show_set', default=False, action='store_true',
                   help='show resulting division of the training set')

args = parser.parse_args()

visualizeFunction(mlp, training_x, training_y)

if args.step_by_step:
    mlp.presentationOfTraining(training_x, training_y)
else:
    mlp.train(training_x, training_y, test_x, test_y, plotLoss = args.plot_loss)

if args.show_set:
    visualizeFunction(mlp, training_x, training_y)
