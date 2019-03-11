import numpy as np

class identity:

    def call(self, x):
        return x

    def derivative(self, x):
        return 1

class softmax:

    def call(self, x):
        e_x = np.exp(x - np.expand_dims(np.max(x, axis = 1), axis = 1))
        assert np.sum(np.expand_dims(e_x.sum(axis=1), axis = 1)) > 0
        return e_x / np.expand_dims(e_x.sum(axis=1), axis = 1)

    def derivative(self, x):
        raise NotImplemented()
        '''
        # this produces some kind of jacobian, what next?
        sm = self.call(x)
        # print('sm', sm)
        summed = np.expand_dims(np.sum(x, axis = 1), axis = 1)
        this gives vector on ones
        jacobian = ???
        # print('summed', summed)
        return - sm * summed + sm this gives 0 in theory 
        '''

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

class crossEntropyWithSoftmax:

    def call(self, pred, y):
        sm = softmax().call(pred)
        print(sm)
        return - np.sum(y * np.log(sm)) / len(y)

    def derivative(self, pred, y):
        sm = softmax().call(pred)
        return (sm - y )/ len(y)

class MSE:

    def call(self, pred, y):
        return 1/len(y) * np.sum((y-pred)**2)

    def derivative(self, pred, y):
        tmp = 1 / len(y) * (pred - y)
        return tmp