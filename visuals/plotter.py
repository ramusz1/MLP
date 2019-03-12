import matplotlib.pyplot as plt
import numpy as np

#fast plotting using blit        
class LossPlotter:
    
    def __init__(self, maxXValue, minY = 0, maxY = 2, bufor = 2):
        self.minY = minY
        self.maxY = maxY
        self.bufor = bufor
        self.maxXValue = maxXValue
        self.fig, self.ax = plt.subplots(1, 1)
        self.drawFigure()
        #always draw 100 times 
        self.draw_moments = np.round(np.linspace(0, maxXValue, 100))
        self.lossList = []

    def drawFigure(self):
        self.ax.set_ylim(self.minY, self.maxY)
        self.ax.set_xlim(-self.maxXValue / 100, self.maxXValue)
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.points = self.ax.plot([], [], 'b')[0]
        self.pointsVal = self.ax.plot([],[],'r')[0]
        self.ax.legend(['Loss on training set','Loss on validation set'])
        self.rangeChanged = False
    
    def plotLive(self, currentMoment, loss):
        self.lossList.append(loss)
        newPointsMin, newPointsMax = min(loss), max(loss)
        if newPointsMin < self.minY:
            self.minY = newPointsMin - self.bufor
            self.rangeChanged = True
        if newPointsMax > self.maxY:
            self.maxY = newPointsMax + self.bufor
            self.rangeChanged = True

        if currentMoment in self.draw_moments:
            self.points.set_data(range(currentMoment+1), np.transpose(self.lossList)[0])
            self.pointsVal.set_data(range(currentMoment+1), np.transpose(self.lossList)[1])
            if self.rangeChanged:
                self.drawFigure()
                self.ax.draw
                self.rangeChanged = False
            else:
                self.fig.canvas.restore_region(self.background)
            # redraw just the points
            self.ax.draw_artist(self.points)
            self.ax.draw_artist(self.pointsVal)
            plt.pause(0.00001)
            # fill in the axes rectangle
            self.fig.canvas.blit(self.ax.bbox)