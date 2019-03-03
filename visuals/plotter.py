import matplotlib.pyplot as plt
import numpy as np

#fast plotting using blit        
class LossPlotter:
    
    def __init__(self, maxXValue):
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_ylim(0, 5) 
        self.ax.set_xlim(0, maxXValue)      
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.points = self.ax.plot([], [], '-')[0]
        #always draw 100 times 
        self.draw_moments = np.round(np.linspace(0, maxXValue, 100))
        self.lossList = []
    
    def plotLive(self, currentMoment, loss):
        self.lossList.append(loss)
        if currentMoment in self.draw_moments:
            self.points.set_data(range(currentMoment+1), self.lossList)
            self.fig.canvas.restore_region(self.background)
            # redraw just the points
            self.ax.draw_artist(self.points)
            plt.pause(0.00001)
            # fill in the axes rectangle
            self.fig.canvas.blit(self.ax.bbox)