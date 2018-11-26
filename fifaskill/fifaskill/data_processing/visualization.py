import numpy as np
from matplotlib import pyplot as plt


def plot_loss(*models):
    colors = ['red', 'green', 'blue', 'purple']
    plt.figure()
    for i in range(len(models)):
        model = models[i]
        plt.plot(np.arange(model.loss), model.loss, color=colors[i])
    plt.show()