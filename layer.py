import numpy as np

class Layer:

    # forward pass with gradient saving
    def forwardWithSave(self, x):
        self.input = x
        return self.forward(x)
        
    # backprop
    def backprop(self, gradIn):
        raise NotImplemented

    def forward(self, input):
        raise NotImplemented


class FullyConnected(Layer):

    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.weight = np.random.randn(inputSize, outputSize)
        self.weightMomentum = np.zeros((inputSize, outputSize))
        self.bias = np.random.randn(outputSize)
        self.biasMomentum = np.zeros(outputSize)

    def forward(self, input):
        return np.matmul(input, self.weight) + self.bias

    # eta : momentumMultiplier
    def backprop(self, gradIn, learningRate, eta):
        # calculate deltas for bias and weight
        biasDelta = np.sum(gradIn, axis = 0)
        self.updateBias(biasDelta)
        weightDelta = np.matmul(self.input.T, gradIn)
        self.updateWeight(weightDelta)
        return np.matmul(gradIn, self.weight.T)

    # momentum from
    # https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
    def updateWeight(self, weightDelta):
        self.weightMomentum = eta * self.weightMomentum + (1 - eta) * weightDelta
        self.weight -= self.learningRate * self.weightMomentum

    def updateBias(self, biasDelta):
        self.biasMomentum = eta * self.biasMomentum + (1 - eta) * biasDelta
        self.bias -= self.learningRate * self.biasMomentum

    
class Activation(Layer):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func.call(x)

    def backprop(self, gradIn, learningRate, eta):
        return self.func.derivative(self.input) * gradIn

class Loss(Layer):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forwardWithSave(self, prediction, y):
        self.prediction = prediction
        self.y = y
        return self.forward(prediction, y)

    def forward(self, prediction, y):
        return self.func.call(prediction, y)

    def backprop(self, gradIn, learningRate, eta):
        return self.func.derivative(self.prediction, self.y) * gradIn
