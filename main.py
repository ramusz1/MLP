import numpy as np
from sklearn import datasets
from mlp import MLP
import pandas as pd
from visuals.setVisualization import visualizeSet

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

# prepare datasets
#x, y = loadIris()
x, y = loadDataset('datasets/classification/data.simple.test.1000.csv')
ind  = np.arange(len(x))
np.random.shuffle(ind)

# iris uses labels starting from 0, downloaded datasets use labels starting from 1
# it's problematic in class maping later on
y = y - np.min(y)

inputSize = x.shape[1]
outputSize = len(np.unique(y))

mlp = MLP([inputSize, 8, outputSize], usesBias = True)

trainingSize = int(0.8 * len(x))

training_x, test_x = x[ind[:trainingSize]], x[ind[trainingSize:]]
training_y, test_y = y[ind[:trainingSize]], y[ind[trainingSize:]]

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
    mlp.presentationOfTraining(training_x, training_y)
else:
    mlp.train(training_x, training_y, plotLoss = args.plot_loss)
    print('Accuracy on training set: ', mlp.accuracy(training_x, training_y))
    print('Accuracy on test set: ', mlp.accuracy(test_x,test_y))

if args.show_set:
    if x.shape[1] != 2:
        print('set visualization is only possible wneh there are 2 features')
    visualizeSet(mlp,x,y)

    pass


