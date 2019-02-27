import numpy as np
from sklearn import datasets

class MLP:

    def __init__(self, layers, alpha = 0.01, batch_size = 4, max_iter = 100):
        self.layers = layers
        self.layersCount = len(self.layers)
        self.__initWeights()
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_iter = max_iter
        

    def __initWeights(self):
        self.weights = []
        for i in range(len(self.layers) - 1):
            self.weights.append( 
                np.random.rand(self.layers[i+1], self.layers[i]))
    
    # train rozdzielony na 2 funkcje 
    def train(self, x, y):
        for i in range(self.max_iter):
            loss = self.trainEpoch(x,y)
            #print(loss)
        
    def trainEpoch(self, x, y):
        for i in range(0, np.shape(x)[1], self.batch_size):
            X, Y = x[:,i:i+self.batch_size], y[i:i+self.batch_size]
            Y = self.mapClasses(Y) 
            a, z = self.forward(X)
            pred = a[-1]
            loss = self.loss(pred, Y)
            delta = self.backprop(a, z, Y)
            self.updateWeights(delta)
        return loss

    # z wektora robi macierz jedynek
    def mapClasses(self, y):
        y1 = np.zeros([self.layers[-1], np.shape(y)[0]])
        for i in range(self.layers[-1]):
            ind = np.where(y == i)
            y1[(i,ind[0])] = 1
        return y1

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
 
    def backprop(self, a, z, y):
        lossDerivative = self.getLossDerivative(a[-1], y)
#        self.debug(lossDerivative, 'loss')
        derivativeChain = lossDerivative
#        self.debug(derivativeChain, 'dc')
        deltas = []
        for l in range(1, self.layersCount-1):
            delta = np.matmul(derivativeChain, a[-l-1].T)
#            self.debug(delta, 'delta')
            deltas.append(delta)
            derivativeChain = np.matmul(self.weights[-l].T, derivativeChain) * self.activationDx(z[-l-1])
#            self.debug(derivativeChain, 'dc')

        delta = np.matmul(derivativeChain, a[0].T)
#        self.debug(delta, 'delta')
        deltas.append(delta)   
#        for d in deltas:
#            print(d.shape)
#        for w in self.weights:
#            print(w.shape)
        return deltas 

    def getLossDerivative(self, a, y):
        return a - y
    
    def updateWeights(self, delta):
        for i in range(self.layersCount-1): #warstw jest o 1 wiecej niż wag
            self.weights[i] = self.weights[i] - self.alpha * delta[-i-1]      
    
    #forward bez zapisywania, wybierana jest klasa z największym prawd.
    def predict(self, x):
        for w in self.weights:
            z = np.matmul(w, x)
            x = self.activation(z)
        out = np.argmax(x)
        return out
    
    #testowa metoda do sprawdzenia, czy sieć się uczy       
    def accuracy(self,x,y):
        suma = 0
        n = np.shape(x)[1] #ile wektorów
        for i in range(n):
            out = self.predict(x[:,i])
            suma = suma + (out==y[i])
        return suma/n
    
   
    @staticmethod
    def debug(x, name):
        print(name, x.shape)


#mlp = MLP([4,5,100,123,4,612,6,2])
#x = np.arange(4)
#xMat = np.arange(12).reshape(4,3)
## y to one hot
#y = np.array([[0,1,0], 
#              [1,0,0]])
#print(y)
#print(xMat)
#print(mlp.weights[0].shape, mlp.weights[1].shape)
#output = mlp.train(xMat, y)

data = datasets.load_iris()
x = data['data']
y = data['target']
y = y[:, np.newaxis]
ind  = list(range(len(x)))
np.random.shuffle(ind)

mlp = MLP([4,8,3])

# transponowanie x, żeby były kolumnowe
training_x, test_x = x[ind[:120]].T, x[ind[120:]].T
training_y, test_y = y[ind[:120]], y[ind[120:]]

mlp.train(training_x, training_y)

print('Accuracy on test set: ' + str(mlp.accuracy(test_x,test_y)[0]))
print('Accuracy on training set: ' + str(mlp.accuracy(training_x,training_y)[0]))
