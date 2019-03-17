from mlp import MLP
import functions as fn
from layer import *

# change the values of these variables to set used datasets
trainDataset = 'datasets/classification/data.three_gauss.train.100.csv'
testDataset = 'datasets/classification/data.three_gauss.test.100.csv'

# you can freely change parameters of the MLP contructor
inputSize = 2
hiddenSize = 16
outputSize = 3
mlp = MLP(
    layers = [
        FullyConnected(inputSize, hiddenSize),
        Activation(fn.relu()),
        FullyConnected(hiddenSize, hiddenSize),
        Activation(fn.relu()),
        FullyConnected(hiddenSize, hiddenSize),
        Activation(fn.relu()),
        FullyConnected(hiddenSize, outputSize)
    ],
    lossFunction = Loss(fn.crossEntropyWithSoftmax()),
    learningRate = 0.05,
    lrDecay = 0.97,
    eta = 0.1,
    batchSize = 36,
    maxIter = 900
)