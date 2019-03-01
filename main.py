import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
#from Graph import Graph 

class MLP:

    def __init__(self, layers, alpha = 0.9, batch_size = 2, max_iter =300):
        self.layers = layers
        self.layersCount = len(self.layers)
        self.__initWeights()
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.momentum = np.zeros(len(self.weights))
        

    def __initWeights(self):
        self.weights = []
        for i in range(len(self.layers) - 1):
            self.weights.append( 
                np.random.rand(self.layers[i], self.layers[i+1]))
    
    # train rozdzielony na 2 funkcje 
    def train(self, x, y, plotLoss = False, drawGraph = True):
        if plotLoss:
            fig, ax, points, bg, d_m = self.initPlot()
            lossList = []
            for i in range(self.max_iter):
                loss = self.trainEpoch(x,y)
                lossList.append(loss)
                self.plotLossLive(i, lossList, points, fig, ax, bg, d_m)
#        else:
#            for i in range(self.max_iter):
#                loss = self.trainEpoch(x,y)
#                print(loss)
        if drawGraph:
            fig = self.initGraph()
            
            for i in range(self.max_iter):
                loss = self.trainEpoch(x,y)
                print(loss)
                if(i%2 == 0):
                    ax = self.drawNodes(fig)
                    self.drawEdges(ax)
                    ax.clear()
        
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
            self.weights[i] = self.weights[i] - self.alpha * (delta[-i-1] + self.momentum[-i-1]) 
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
    
#    fast plotting using blit        
    def initPlot(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_ylim(0, 5) 
        ax.set_xlim(0, self.max_iter)      
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(ax.bbox)
        points = ax.plot([], [], '-')[0]
        #always draw 100 times 
        draw_moments = np.round(np.linspace(0,self.max_iter,100))
        return fig, ax, points, background, draw_moments
    
    def plotLossLive(self, k ,loss, points, fig, ax, bg, draw_moments):
        if k in draw_moments:
            points.set_data(range(k+1), loss)
            fig.canvas.restore_region(bg)
            # redraw just the points
            ax.draw_artist(points)
            plt.pause(0.00001)
            # fill in the axes rectangle
            fig.canvas.blit(ax.bbox)
            
    """ 
    modified function from
    https://gist.github.com/craffel/2d727968c3aaebd10359
    if weight < 0 plots red line else black
    (for plotting) weights scaled to (-1,1), beacause opacity parameter (alpha) takes values (0,1) 
    
    """
    def initGraph(self):
        plt.close()
        fig = plt.figure(figsize=(12, 12))
        return fig
    
    def drawNodes(self, ax):
        left, right, bottom, top = .1, .9, .1, .9
        v_spacing = (top - bottom)/float(max(self.layers))
        h_spacing = (right - left)/float(len(self.layers) - 1)
        # Nodes
        for n, layer_size in enumerate(self.layers):
            layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
            for m in range(layer_size):
                circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
                ax.add_artist(circle)
        return ax   
     
    def drawEdges(self, ax):
        left, right, bottom, top = .1, .9, .1, .9
        v_spacing = (top - bottom)/float(max(self.layers))
        h_spacing = (right - left)/float(len(self.layers) - 1)
        for n, (layer_size_a, layer_size_b,) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
            alfa  = self.weights[n] 
            alfa[alfa>1] = 1    
            alfa[alfa<-1] = -1
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    w = alfa[m,o]
                    line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                      [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c=self.if_color(w), alpha=np.abs(w))
                    ax.add_artist(line)
        plt.pause(0.001)
        
    @staticmethod
    def if_color(x):
        if x<0:
            return 'r'
        if x>=0:
            return 'k'      
                
data = datasets.load_iris()
x = data['data']
y = data['target']
ind  = np.arange(len(x))
np.random.shuffle(ind)

mlp = MLP([4,8,4,3])

training_x, test_x = x[ind[:120]], x[ind[120:]]
training_y, test_y = y[ind[:120]], y[ind[120:]]
mlp.train(training_x, training_y)

print('Accuracy on training set: ', mlp.accuracy(training_x,training_y))
print('Accuracy on test set: ', mlp.accuracy(test_x,test_y))
