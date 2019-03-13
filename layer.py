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
        print('biasGrad', biasDelta)
        self.updateBias(biasDelta, learningRate, eta)
        weightDelta = np.matmul(self.input.T, gradIn)
        print('weightGrad', weightDelta)
        grad_out = np.matmul(gradIn, self.weight.T)
        self.updateWeight(weightDelta, learningRate, eta)
        print('xGrad', grad_out)
        return grad_out

    # momentum from
    # https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
    def updateWeight(self, weightDelta, learningRate, eta):
        self.weightMomentum = eta * self.weightMomentum + (1 - eta) * weightDelta
        self.weight -= learningRate * self.weightMomentum

    def updateBias(self, biasDelta, learningRate, eta):
        self.biasMomentum = eta * self.biasMomentum + (1 - eta) * biasDelta
        self.bias -= learningRate * self.biasMomentum

    
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
