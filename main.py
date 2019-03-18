import numpy as np
from sklearn import datasets
from mlp import MLP
import pandas as pd
from visuals.setVisualization import visualizeSet
from gridSearch import runGridSearch

import functions as fn
from layer import *
import config

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

training_x, training_y = loadDataset(config.trainDataset)
test_x, test_y = loadDataset(config.testDataset)

training_y = training_y - np.min(training_y)
test_y = test_y - np.min(test_y)
inputSize = training_x.shape[1]
outputSize = len(np.unique(training_y))

training_y_one_hot = mapClasses(training_y, outputSize)
test_y_one_hot = mapClasses(test_y, outputSize)

import argparse

parser = argparse.ArgumentParser(description='Run mlp classifier training.')

parser.add_argument('--step_by_step', default=False, action='store_true',
                   help='step by step mode with neural network graph visualization')

parser.add_argument('--plot_loss', default=False, action='store_true',
                   help='show live plot of the loss')

parser.add_argument('--show_set', default=False, action='store_true',
                   help='show resulting division of the training set')

parser.add_argument('--grid_search', default=False, action='store_true',
                   help='run grid search')

args = parser.parse_args()

if args.grid_search:
    runGridSearch(training_x, training_y_one_hot, training_y, test_x, test_y_one_hot, test_y, inputSize, outputSize, Loss(fn.crossEntropyWithSoftmax()))
    exit(0)

if args.step_by_step:
    config.mlp.presentationOfTraining(training_x, training_y_one_hot)
else:
    config.mlp.train(training_x, training_y_one_hot, test_x, test_y_one_hot, plotLoss = args.plot_loss)

print('Accuracy on training set: ', config.mlp.accuracy(training_x, training_y))
print('Accuracy on test set: ', config.mlp.accuracy(test_x,test_y))

if args.show_set:
    if training_x.shape[1] != 2:
        print('set visualization is only possible when there are 2 features')
    visualizeSet(config.mlp, training_x, training_y)
