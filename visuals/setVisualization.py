import matplotlib.pyplot as plt
import numpy as np


def visualizeSet(model, data, label):
    plt.clf()
    x, y = data[:,0], data[:,1]
    xRange = np.min(x), np.max(x)
    yRange = np.min(y), np.max(y)
    resolution = 100
    spaceDivision = getDividedSpace(model, xRange, yRange, resolution)
    
    origin = (xRange[0] * resolution, yRange[0] * resolution)
    plt.imshow(spaceDivision,
        cmap = 'Set1',
        origin = 'lower',
        alpha = 0.5, 
        extent = [xRange[0], xRange[1], yRange[0], yRange[1]])
    plt.scatter(x, y, c = label, cmap = 'Set1')

    plt.show()

def getDividedSpace(model, xRange, yRange, resolution):
    width = int((xRange[1] - xRange[0]) * resolution)
    height = int((yRange[1] - yRange[0]) * resolution)
    minX, minY = xRange[0], yRange[0]
    image = np.zeros(shape=(height, width))
    for y, row in enumerate(image):
        for x, _ in enumerate(row):
            image[y][x] = model.predictLabel(
                [[x/resolution + minX, y/resolution + minY]])

    return image

