import matplotlib.pyplot as plt

def visualizeFunction(model, x, y):
    plt.figure(41)
    plt.plot(x, y, '.')
    plt.plot(x, model.predict(x), '.')
    plt.show()