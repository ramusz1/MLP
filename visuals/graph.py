import matplotlib.pyplot as plt
import numpy as np

""" 
modified code from
https://gist.github.com/craffel/2d727968c3aaebd10359
if weight < 0 plots red line else black
(for plotting) weights scaled to (-1,1), beacause opacity parameter (alpha) takes values (0,1) 
"""

class NetworkGraph:
    def __init__(self, network):
        self.network = network
        plt.close() # ??? 
        self.fig = plt.figure(figsize=(12, 12))
        self.ax = self.fig.gca()
    
    def draw(self):
        self.drawNodes()
        self.drawEdges()
        self.ax.clear()

    def drawNodes(self):
        left, right, bottom, top = .1, .9, .1, .9
        v_spacing = (top - bottom)/max(self.network.layers)
        h_spacing = (right - left)/(len(self.network.layers) - 1)
        # Nodes
        for n, layer_size in enumerate(self.network.layers):
            layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2
            for m in range(layer_size):
                circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4,
                                color='w', ec='k', zorder=4)
                self.ax.add_artist(circle)
     
    def drawEdges(self):
        left, right, bottom, top = .1, .9, .1, .9
        v_spacing = (top - bottom)/max(self.network.layers)
        h_spacing = (right - left)/(len(self.network.layers) - 1)
        for n, (layer_size_a, layer_size_b,) in enumerate(zip(self.network.layers[:-1], self.network.layers[1:])):
            layer_top_a = v_spacing*(layer_size_a - 1)/2 + (top + bottom)/2
            layer_top_b = v_spacing*(layer_size_b - 1)/2 + (top + bottom)/2
            alfa  = self.network.weights[n]
            # modified sigmoid : R -> (-1,1)
            alfa = 2 / (1 + np.exp(-alfa)) - 1
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    w = alfa[m,o]
                    line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                      [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c=self.if_color(w), alpha=np.abs(w))
                    self.ax.add_artist(line)
        plt.pause(0.001)
        
    @staticmethod
    def if_color(x):
        if x<0:
            return 'r'
        if x>=0:
            return 'b'  