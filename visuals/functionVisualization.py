import matplotlib.pyplot as plt

def visualizeFunction(model, x, y):
    fig, ax = plt.subplots(1,1)
    ax.plot(x[:,0], y, '.')
    ax.plot(x[:,0], model.predict(x), '.')
    ax.legend(['Test set','Predicted values'])
    plt.show()