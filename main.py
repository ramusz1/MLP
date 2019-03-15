import numpy as np
from sklearn import datasets
from mlp import MLP
import pandas as pd
from visuals.setVisualization import visualizeSet

import functions as fn
from layer import *

def loadIris():
    data = datasets.load_iris()
    x = data['data']
    y = data['target']
    return x, y 


def loadDataset(filename):
    dataset = pd.read_csv(filename)
    x = dataset.drop(columns = ['cls']).values
    y = dataset['cls'].values
    return x, y 

# z wektora robi macierz jedynek
def mapClasses(y, yRange):
    oneHotVectors = np.zeros((len(y), yRange))
    oneHotVectors[np.arange(len(y)), y] = 1
    return oneHotVectors

training_x, training_y = loadDataset('datasets/classification/data.three_gauss.train.100.csv')
test_x, test_y = loadDataset('datasets/classification/data.three_gauss.test.100.csv')

training_y = training_y - np.min(training_y)
test_y = test_y - np.min(test_y)
inputSize = training_x.shape[1]
outputSize = len(np.unique(training_y))

training_y_one_hot = mapClasses(training_y, outputSize)
test_y_one_hot = mapClasses(test_y, outputSize)

mlp = MLP(
    layers = [
        FullyConnected(inputSize, 8),
        Activation(fn.sigmoid()),
        FullyConnected(8, outputSize),
    ],
    lossFunction = Loss(fn.crossEntropyWithSoftmax()),
    learningRate = 0.1,
    lrDecay = 1,
    eta = 0.00,
    batchSize = 64,
    maxIter = 700
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


if args.step_by_step:
    mlp.presentationOfTraining(training_x, training_y_one_hot)
else:
    mlp.train(training_x, training_y_one_hot, test_x, test_y_one_hot, plotLoss = args.plot_loss)
    print('Accuracy on training set: ', mlp.accuracy(training_x, training_y))
    print('Accuracy on test set: ', mlp.accuracy(test_x,test_y))

if args.show_set:
    if training_x.shape[1] != 2:
        print('set visualization is only possible when there are 2 features')
    visualizeSet(mlp, training_x, training_y)
