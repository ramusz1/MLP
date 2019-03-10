import numpy as np

class softmax:

    def call(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.expand_dims(e_x.sum(axis=1), axis = 1)

    def derivative(self, x):
        # 50% sure this is good
        sm = self.call(x)
        summed = np.expand_dims(np.sum(x, axis = 1), axis = 1)
        return - sm * summed + sm

class sigmoid:

    def call(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.call(x) * (1 - self.call(x))

class relu:

    def call(self, x):
        # x[x < 0 ] = 0, faster but inplace - no good
        return np.maximum(x, 0)

    def derivative(self, x):
        return (x > 0).astype(int)

class crossEntropy:

    def call(self, pred, y):
        return - np.sum(y * np.log(pred))

    def derivative(self, pred, y):
        return - y / pred

class MSE:

    def call(self, pred, y):
        return 1/len(y) * np.sum((y-pred)**2)

    def derivative(self, pred, y):
        return (pred - y) / len(y)