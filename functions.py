import numpy as np

class softmax:

    def call(self, x):
        e_x = np.exp(x - np.expand_dims(np.max(x, axis = 1), axis = 1))
        assert np.sum(np.expand_dims(e_x.sum(axis=1), axis = 1)) > 0
        return e_x / np.expand_dims(e_x.sum(axis=1), axis = 1)

    def derivative(self, x):
        raise NotImplemented()

class sigmoid:

    def call(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        # return 1
        return self.call(x) * (1 - self.call(x))

class relu:

    def call(self, x):
        # x[x < 0 ] = 0, faster but inplace - no good
        return np.maximum(x, 0)

    def derivative(self, x):
        return (x > 0).astype(int)

class crossEntropyWithSoftmax:

    def call(self, pred, y):
        sm = softmax().call(pred)
        return - np.sum(y * np.log(sm)) / len(y)

    def derivative(self, pred, y):
        sm = softmax().call(pred)
        return (sm - y)/ len(y)

class MSE:

    def call(self, pred, y):
        return 1/pred.shape[1] * np.sum((y-pred)**2)

    def derivative(self, pred, y):
        return 2/pred.shape[1] * (pred - y)