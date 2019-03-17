from mlp import MLP
import functions as fn
from layer import *

# change the values of these variables to set used datasets
trainDataset = 'datasets/regression/data.cube.train.100.csv'
testDataset = 'datasets/regression/data.cube.test.100.csv'

# you can freely change parameters of the MLP contructor
mlp = MLP(
    layers = [
        FullyConnected(1, 16),
        Activation(fn.sigmoid()),
        FullyConnected(16, 1),
    ],
    lossFunction = Loss(fn.MSE()),
    learningRate = 0.01,
    lrDecay = 0.99,
    eta = 0.05,
    batchSize = 64,
    maxIter = 1000,
)