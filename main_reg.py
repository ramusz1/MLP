import numpy as np
from sklearn import datasets
from mlp import MLP
import pandas as pd
from visuals.setVisualization import visualizeSet
from visuals.functionVisualization import visualizeFunction
from gridSearch import runGridSearch
import functions as fn
from layer import *

import config_reg

def loadDataset(filename):
    dataset = pd.read_csv(filename)
    x = dataset['x'].values
    y = dataset['y'].values
    x = np.expand_dims(x, axis = 1)
    y = np.expand_dims(y, axis = 1)
    return x, y 

training_x, training_y = loadDataset(config_reg.trainDataset)
test_x, test_y = loadDataset(config_reg.testDataset)

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

if args.step_by_step:
    config_reg.mlp.presentationOfTraining(training_x, training_y)
else:
    config_reg.mlp.train(training_x, training_y, test_x, test_y, plotLoss = args.plot_loss)

if args.show_set:
    visualizeFunction(config_reg.mlp, test_x, test_y)
