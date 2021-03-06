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
            minAlpha = 0.000001,
            lrDecay = 1,
            learningRateUpdateTiming = 10,
            eta = 0.05,
            gamma = 0.99,
            batchSize = 32,
            maxIter = 500,
            earlyStop = True):

        self.learningRate = learningRate
        self.eta = eta
        self.earlyStopIsEnabled = earlyStop
        self.layers = layers
        self.lrDecay = lrDecay
        self.maxIter = maxIter
        self.batchSize = batchSize
        self.lossFunction = lossFunction
        self.learningRateUpdateTiming = learningRateUpdateTiming
        self.minAlpha = minAlpha
        self.bestLayer = None
        self.worseEpochs = 0


    def presentationOfTraining(self, x, y):
        print("presentation mode, press enter to go to next epoch results")
        fig = NetworkGraph(self)
        fig.draw()
        for i in range(self.maxIter):
            self.trainEpoch(x, y)
            fig.draw()
            input('epoch: {}'.format(i))

    def train(self, x, y, xVal, yVal, plotLoss = False):
        self.bestLayer = self.layers, None
        if plotLoss:
            lossPlot = LossPlotter(self.maxIter)
        for i in range(self.maxIter):
            loss = self.trainEpoch(x, y)
            loss_val = self.getLoss(xVal, yVal)
            if plotLoss:
                lossPlot.plotLive(i, [loss,loss_val])
            self.updateBest(loss_val)              
            if self.earlyStopIsEnabled and self.earlyStop(loss_val):
                break  

            self.updateLearningRate(i)
            #sys.stdout.write("\r Learning progress: %d%%" % np.round(i/self.maxIter*100))
            #sys.stdout.flush()
        #print('')

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

    def getDrawable(self):
        fullyConnected = list(filter(self.isDrawableLayer, self.layers))
        layersWidth = list(map(lambda fc: fc.weights.shape[0], fullyConnected))
        layersWidth.append(fullyConnected[-1].weights.shape[1]) # that last layer
        allWeights = list(map(lambda fc: fc.weights, fullyConnected))
        return layersWidth, allWeights

    @staticmethod
    def isDrawableLayer(layer):
        return type(layer) is FullyConnected or type(layer) is FullyConnectedWithoutBias
    
    def updateBest(self,loss):
        if(self.bestLayer[1] == None or self.bestLayer[1]>loss):
            self.bestLayer = self.layers, loss
    
    def earlyStop(self, curr_loss):
        if self.bestLayer[1] < curr_loss:
            self.worseEpochs += 1
        else :
            self.worseEpochs = 0

        if self.worseEpochs > 20:
            self.layers = self.bestLayer[0]
            print('early stop')
            return True
        return False
