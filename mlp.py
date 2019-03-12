import numpy as np

from visuals.plotter import LossPlotter
from visuals.graph import NetworkGraph
import sys

import functions as fn
from layer import * 

class MLP:

    # eta - momentumMultipier
    def __init__(self, 
            layers,
            lossFunction = Loss(fn.crossEntropyWithSoftmax()),
            learningRate = 0.1,
            lrDecay = 1,
            eta = 0.05,
            gamma = 0.99,
            batchSize = 32,
            maxIter = 500):

        self.learningRate = learningRate
        self.eta = eta
        self.layers = layers
        self.lrDecay = lrDecay
        self.maxIter = maxIter
        self.batchSize = batchSize
        self.lossFunction = lossFunction
        self.learningRateUpdateTiming = 10
        self.minAlpha = 0.000001

    def train(self, x, y, xVal, yVal, plotLoss = False):
        if plotLoss:
            lossPlot = LossPlotter(self.maxIter)
        for i in range(self.maxIter):
            loss = self.trainEpoch(x, y)
            loss_val = self.getLoss(xVal, yVal)
            if plotLoss:
                lossPlot.plotLive(i, [loss,loss_val])
            self.updateLearningRate(i)
            sys.stdout.write("\r Learning progress: %d%%" % np.round(i/self.maxIter*100))
            sys.stdout.flush()
        print('')

    # returns loss of last training session
    def trainEpoch(self, x, y):
        ind = np.random.permutation(np.arange(len(x)))
        x, y = x[ind], y[ind]
        for i in range(0, len(x), self.batchSize):
            X, Y = x[i:i+self.batchSize], y[i:i+self.batchSize]
            X = self.trainingForward(X)
            self.lossFunction.forwardWithSave(X, Y)
            self.backprop()
        return self.getLoss(x, y)

    def trainingForward(self, x):
        for l in self.layers:
            x = l.forwardWithSave(x)
        return x

    def backprop(self):
        grad = self.lossFunction.backprop(1, self.learningRate, self.eta)
        for l in reversed(self.layers):
            grad = l.backprop(grad, self.learningRate, self.eta)

    def predict(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def getLoss(self, x, y):
        pred = self.predict(x)
        loss = self.lossFunction.forward(pred, y)
        return loss

    def updateLearningRate(self, epoch):
        if self.learningRate > self.minAlpha and epoch % self.learningRateUpdateTiming == 0:
            self.learningRate *= self.lrDecay

    #forward bez zapisywania, wybierana jest klasa z największym prawd.
    def predictLabel(self, x):
        x = self.predict(x)
        return np.argmax(x, axis = 1)

    def accuracy(self, X, Y):
        prediction = self.predictLabel(X)
        suma = sum(prediction == Y)
        n = len(Y)
        return suma/n
