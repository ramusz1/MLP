import numpy as np

from visuals.plotter import LossPlotter
from visuals.graph import NetworkGraph
import sys


class MLP:

    def __init__(self, layers, usesBias = False, alpha = 0.01, eta = 0.9, batchSize = 32, maxIter = 1000):
        self.layers = layers
        self.layersCount = len(self.layers)
        self.usesBias = usesBias
        self.alpha = alpha
        self.eta = eta
        self.batchSize = batchSize
        self.maxIter = maxIter
        self.__initWeights()
        self.__initMomentum()
        if self.usesBias:
            self.__initBias()

    def __initWeights(self):
        self.weights = []
        for i in range(len(self.layers) - 1):
            self.weights.append( 
                np.random.randn(self.layers[i], self.layers[i+1]))

    def __initMomentum(self):
        self.momentum = []
        for i in range(len(self.layers) -1):
            self.momentum.append(
                np.zeros((self.layers[i], self.layers[i+1])))
        if self.usesBias:
            self.biasMomentum = []
            for i in range(len(self.layers) - 1):
                self.biasMomentum.append(np.random.randn(self.layers[i+1]))

    def __initBias(self):
        self.bias = []
        for i in range(len(self.layers) - 1):
            self.bias.append(np.random.randn(self.layers[i+1]))

    
    def presentationOfTraining(self, x, y):
        print("presentation mode, press enter to go to next epoch results")
        fig = NetworkGraph(self)
        oneHotY = self.mapClasses(y)
        for i in range(self.maxIter):
            self.trainEpoch(x,oneHotY)
            fig.draw()
            input('epoch: {}'.format(i+1))

    def train(self, x, y, plotLoss = False):
        x, y, x_val, y_val = self.makeValidationSets(x,y)
        oneHotY = self.mapClasses(y)
        oneHotYVal = self.mapClasses(y_val)
        
        if plotLoss:
            lossPlot = LossPlotter(self.maxIter)
        for i in range(self.maxIter):
            loss = self.trainEpoch(x,oneHotY)
            pred_val = self.predict(x_val)
            loss_val = self.loss(pred_val,oneHotYVal)
            if plotLoss:
                lossPlot.plotLive(i,[loss,loss_val])
            sys.stdout.write("\r Learning progress: %d%%" % np.round(i/self.maxIter*100))
            sys.stdout.flush()
        print('')
            
    def makeValidationSets(self, x, y, setSize = 0.2):
        n = len(x)
        ind = np.random.choice(range(n), int(np.round(n*setSize)))
        x_val, y_val = x[ind], y[ind]
        x, y = np.delete(x, ind, axis=0), np.delete(y, ind, axis=0)
        return x, y, x_val, y_val
        
    # z wektora robi macierz jedynek
    def mapClasses(self, y):
        oneHotVectors = np.zeros((len(y), self.layers[-1]))
        oneHotVectors[np.arange(len(y)), y] = 1
        return oneHotVectors

    # returns loss of last training session
    def trainEpoch(self, x, y):
        for i in range(0, len(x), self.batchSize):
            X, Y = x[i:i+self.batchSize], y[i:i+self.batchSize]
            a, z = self.forward(X)
            pred = a[-1]
            ## wrong: Displaying loss only on last mini-batch
#           loss = self.loss(pred, Y)
            weightDeltas, biasDeltas = self.backprop(a, z, Y)
            self.updateWeights(weightDeltas, biasDeltas)
        # Fixed: after training on mini-batches we count total loss on whole set
        pred = self.predict(x)
        loss = self.loss(pred, y)
        return loss


    def forward(self, x):
        aList = []
        zList = []
        for i in range(self.layersCount-1):
            aList.append(x)
            z = np.matmul(x, self.weights[i])
            if self.usesBias:
                z = z + self.bias[i]
            x = self.activation(z)
            zList.append(z)

        aList.append(x)
        return aList, zList

    def activation(self, x):
        return 1 / (1 + np.exp(-x))
        
    def activationDx(self, x):
        return self.activation(x) * (1 - self.activation(x))

    # cross entropy
#    def loss(self, pred, y):
#        return - np.sum(y * np.log(pred))
    # MSE
    def loss(self, pred, y):
        return 1/len(y) * np.sum((y-pred)**2)
    
    def backprop(self, a, z, y):
        derivativeChain = self.getLossDerivative(a[-1], y)
        deltas = []
        biasDeltas = []
        for l in range(1, self.layersCount):
            # activation function backprop
            derivativeChain = derivativeChain * self.activationDx(z[-l])
            # addition backprop
            # empty
            # bias gradient is ready
            biasDeltas.append(np.sum(derivativeChain, axis = 0))
            # matrix multiplication X*W backprop
            # gradient for W
            delta = np.matmul(a[-l-1].T, derivativeChain)
            # debug print('W shape' ,self.weights[-l].shape, 'vs', delta.shape)
            deltas.append(delta)
            # gradient for X
            if l < self.layersCount - 1:
                derivativeChain = np.matmul(derivativeChain, self.weights[-l].T)
                # debug print('X shape' , z[-l-1].shape, 'vs', derivativeChain.shape)

        return deltas, biasDeltas

    def getLossDerivative(self, pred, y):
        # imo cross entropy derivative is - y / pred
        # pred - y is mean square error
        return  pred - y
    
    # momentum from :
    # https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
    def updateWeights(self, weightDeltas, biasDeltas):
        for i in range(self.layersCount-1): #warstw jest o 1 wiecej niż wag
            # update momentum
            self.momentum[i] = self.eta * self.momentum[i] + (1 - self.eta) * weightDeltas[-i-1]

            self.weights[i] -= self.alpha * self.momentum[i]
            if self.usesBias:
                self.biasMomentum[i] = self.eta * self.biasMomentum[i] + (1 - self.eta) * biasDeltas[-i-1]
                self.bias[i] -= self.alpha * self.biasMomentum[i]
            # print( 'l2 norm of {}\'th weights delta : {}'.format(i, np.linalg.norm(weightDeltas[-i-1])))

    #forward bez zapisywania, wybierana jest klasa z największym prawd.
    def predictLabel(self, x):
        for w in self.weights:
            z = np.matmul(x, w)
            x = self.activation(z)
        if(x.ndim >1):
            out = np.argmax(x, axis = 1)
        else:
            out = np.argmax(x)
        return out
    
    def predict(self, x):
        for w in self.weights:
            z = np.matmul(x, w)
            x = self.activation(z)
        return x
        
    def accuracy(self, X, Y):
        prediction = self.predictLabel(X)
        suma = sum(prediction == Y)
        n = len(Y)
        return suma/n

