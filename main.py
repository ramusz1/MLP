import numpy as np

class MLP:

    def __init__(self, layers, alfa = 0.01):
        self.layers = layers
        self.layersCount = len(self.layers)
        self.__initWeights()
        self.alfa = alfa

    def __initWeights(self):
        self.weights = []
        for i in range(len(self.layers) - 1):
            self.weights.append( 
                np.random.rand(self.layers[i+1], self.layers[i]))

    def train(self, x, y):
        a, z = self.forward(x)
        pred = a[-1]
        loss = self.loss(pred, y)
        self.backprop(a, z, y, loss)

    def forward(self, x):
        aList = []
        zList = []
        aList.append(x)
        for w in self.weights:
            z = np.matmul(w, x)
            x = self.activation(z)
            aList.append(x)
            zList.append(z)
        return aList, zList

    def activation(self, x):
        return 1 / (1 + np.exp(-x))
        
    def activationDx(self, x):
        return self.activation(x) * (1 - self.activation(x))

    # cross entropy
    def loss(self, pred, y):
        return - np.sum(y * np.log(pred))
 
    def backprop(self, a, z, y, loss):
        lossDerivative = self.getLossDerivative(a[-1], y)
        self.debug(lossDerivative, 'loss')
        derivativeChain = lossDerivative
        self.debug(derivativeChain, 'dc')
        deltas = []
        for l in range(1, self.layersCount-1):
            delta = np.matmul(derivativeChain, a[-l-1].T)
            self.debug(delta, 'delta')
            deltas.append(delta)
            #derivativeChain = np.matmul(derivativeChain, self.weights[-l])
            #derivativeChain = derivativeChain * self.activationDx(z[-l-1])
            self.debug(derivativeChain, 'dc')

        delta = np.matmul(derivativeChain, a[0].T)
        self.debug(delta, 'delta')
        deltas.append(delta)

    def getLossDerivative(self, a, y):
        return a - y

    @staticmethod
    def debug(x, name):
        print(name, x.shape)

mlp = MLP([4,5,2])
x = np.arange(4)
xMat = np.arange(8).reshape(4,2)
# y to one hot
y = np.array([[0,1], [1,0]])
print(y)
print(xMat)
print(mlp.weights[0].shape, mlp.weights[1].shape)
output = mlp.train(xMat, y)







