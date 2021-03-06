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
        
    def __str__(self):
        return ''

    def reset(self):
        return


class FullyConnected(Layer):

    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.reset()

    def reset(self):

        self.weights = np.random.randn(self.inputSize, self.outputSize)
        self.weightsMomentum = np.zeros((self.inputSize, self.outputSize))
        self.bias = np.random.randn(self.outputSize)
        self.biasMomentum = np.zeros(self.outputSize)

    def forward(self, input):
        return np.matmul(input, self.weights) + self.bias

    # eta : momentumMultiplier
    def backprop(self, gradIn, learningRate, eta):
        # calculate deltas for bias and weight
        biasDelta = np.sum(gradIn, axis = 0)
        self.updateBias(biasDelta, learningRate, eta)
        weightDelta = np.matmul(self.input.T, gradIn)
        grad_out = np.matmul(gradIn, self.weights.T)
        self.updateWeight(weightDelta, learningRate, eta)
        return grad_out

    # momentum from
    # https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
    def updateWeight(self, weightDelta, learningRate, eta):
        self.weightsMomentum = eta * self.weightsMomentum + (1 - eta) * weightDelta
        self.weights -= learningRate * self.weightsMomentum

    def updateBias(self, biasDelta, learningRate, eta):
        self.biasMomentum = eta * self.biasMomentum + (1 - eta) * biasDelta
        self.bias -= learningRate * self.biasMomentum
        
    def __str__(self):
        return str(self.weights.shape)


class FullyConnectedWithoutBias(Layer):

    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.reset()

    def reset(self):
        self.weights = np.random.randn(self.inputSize, self.outputSize)
        self.weightsMomentum = np.zeros((self.inputSize, self.outputSize))

    def forward(self, input):
        return np.matmul(input, self.weights)

    # eta : momentumMultiplier
    def backprop(self, gradIn, learningRate, eta):
        # calculate deltas for weights
        weightDelta = np.matmul(self.input.T, gradIn)
        grad_out = np.matmul(gradIn, self.weights.T)
        self.updateWeight(weightDelta, learningRate, eta)
        return grad_out

    # momentum from
    # https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
    def updateWeight(self, weightDelta, learningRate, eta):
        self.weightsMomentum = eta * self.weightsMomentum + (1 - eta) * weightDelta
        self.weights -= learningRate * self.weightsMomentum

    def __str__(self):
        return str(self.weights.shape)
    
class Activation(Layer):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func.call(x)

    def backprop(self, gradIn, learningRate, eta):
        return self.func.derivative(self.input) * gradIn
    
    def __str__(self):
        return str(self.func)

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
    def __str__(self):
        return str(self.func)
