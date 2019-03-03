import numpy as np
from sklearn import datasets

from visuals.plotter import LossPlotter
from visuals.graph import NetworkGraph

class MLP:

    def __init__(self, layers, alpha = 0.01, eta = 0.01, batch_size = 16, epochLength = 300):
        self.layers = layers
        self.layersCount = len(self.layers)
        self.__initWeights()
        self.alpha = alpha
        self.eta = eta
        self.batch_size = batch_size
        self.epochLength = epochLength
        self.momentum = np.zeros(len(self.weights))
        

    def __initWeights(self):
        self.weights = []
        for i in range(len(self.layers) - 1):
            self.weights.append( 
                np.random.rand(self.layers[i], self.layers[i+1]))
    
    # train rozdzielony na 2 funkcje 
    def train(self, x, y, plotLoss = False, drawGraph = False):
        if plotLoss:
            lossPlot = LossPlotter(self.epochLength)
            lossList = []
            for i in range(self.epochLength):
                loss = self.trainEpoch(x,y)
                lossList.append(loss)
                lossPlot.plotLive(i, lossList)

        if drawGraph:
            fig = NetworkGraph(self)
            for i in range(self.epochLength):
                loss = self.trainEpoch(x,y)
                print(loss)
                if(i%2 == 0):
                    fig.draw()
        
    def trainEpoch(self, x, y):
        for i in range(0, len(x), self.batch_size):
            X, Y = x[i:i+self.batch_size], y[i:i+self.batch_size]
            Y = self.mapClasses(Y)
            a, z = self.forward(X)
            pred = a[-1]
            loss = self.loss(pred, Y)
            delta = self.backprop(a, z, Y)
            self.updateWeights(delta)
        return loss

    # z wektora robi macierz jedynek
    def mapClasses(self, y):
        oneHotVectors = np.zeros((len(y), self.layers[-1]))
        oneHotVectors[np.arange(len(y)), y] = 1
        return oneHotVectors

    def forward(self, x):
        aList = []
        zList = []
        aList.append(x)
        for w in self.weights:
            z = np.matmul(x, w)
            x = self.activation(z)
            aList.append(x)
            zList.append(z)
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
        for l in range(1, self.layersCount):
            # activation function backprop
            derivativeChain = derivativeChain * self.activationDx(z[-l])
            # addition backprop
            # empty
            # matrix multiplication X*W backprop
            # gradient for W
            delta = np.matmul(a[-l-1].T, derivativeChain)
            # debug print('W shape' ,self.weights[-l].shape, 'vs', delta.shape)
            deltas.append(delta)
            # gradient for X
            if l < self.layersCount - 1:
                derivativeChain = np.matmul(derivativeChain, self.weights[-l].T)
                # debug print('X shape' , z[-l-1].shape, 'vs', derivativeChain.shape)

        return deltas 

    def getLossDerivative(self, pred, y):
        # imo cross entropy derivative is - y / pred
        # pred - y is mean square error
        return  pred - y
    
    def updateWeights(self, delta):
        for i in range(self.layersCount-1): #warstw jest o 1 wiecej niż wag
            self.weights[i] = self.weights[i] - self.alpha * delta[-i-1] - self.eta * self.momentum[-i-1]
            # print( 'l2 norm of {}\'th weights delta : {}'.format(i, np.linalg.norm(delta[-i-1])))
        self.momentum = delta
 
    #forward bez zapisywania, wybierana jest klasa z największym prawd.
    def predict(self, x):
        for w in self.weights:
            z = np.matmul(x, w)
            x = self.activation(z)
        out = np.argmax(x)
        return out
    
    def accuracy(self, X, Y):
        suma = 0
        for x, y in zip(X,Y):
            prediction = self.predict(x)
            suma = suma + (prediction==y)
        n = len(Y)
        return suma/n
                
data = datasets.load_iris()
x = data['data']
y = data['target']
ind  = np.arange(len(x))
np.random.shuffle(ind)

mlp = MLP([4,8,3])

training_x, test_x = x[ind[:120]], x[ind[120:]]
training_y, test_y = y[ind[:120]], y[ind[120:]]
mlp.train(training_x, training_y, plotLoss = False, drawGraph = True)

print('Accuracy on training set: ', mlp.accuracy(training_x,training_y))
print('Accuracy on test set: ', mlp.accuracy(test_x,test_y))
