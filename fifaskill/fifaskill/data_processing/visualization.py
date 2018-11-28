import numpy as np
from matplotlib import pyplot as plt


def plot_loss(*models):
    colors = ['red', 'green', 'blue', 'purple']
    plt.figure()
    for i in range(len(models)):
        model = models[i]
        plt.plot(np.arange(len(model.loss)), model.loss, c=colors[i])
    plt.show()


def plot_ppcs(*models, log_plot=False):

    for i in range(len(models)):
        plt.figure()
        model = models[i]
        ppc = model.ppc('mean')

        if type(ppc) is tuple:
            _, axes = plt.subplots(4, 1, sharex=True)
            if log_plot:
                min_val = np.min([np.min(ppc[0]), np.min(ppc[1]), 
                                  np.min(model.ys[:, 0]), 
                                  np.min(model.ys[:, 0])])
                max_val = np.max([np.max(ppc[0]), np.max(ppc[1]), 
                                  np.max(model.ys[:, 0]), 
                                  np.max(model.ys[:, 0])])
                logbins = np.geomspace(min_val, max_val, 8)
                axes[0].hist(ppc[0], bins=logbins)
                axes[0].axvline(np.mean(model.ys[:, 0]), color='red')
                axes[1].hist(model.ys[:, 0], color='red', bins=logbins)
                axes[1].axvline(np.mean(model.ys[:, 0]), color='red')

                axes[2].hist(ppc[1], bins=logbins)
                axes[2].axvline(np.mean(model.ys[:, 1]), color='red')
                axes[3].hist(model.ys[:, 1], color='red', bins=logbins)
                axes[3].axvline(np.mean(model.ys[:, 1]), color='red')
                axes[3].set_xscale('log')
                axes[2].set_xscale('log')
                axes[1].set_xscale('log')
                axes[0].set_xscale('log')
            else:
                axes[0].hist(ppc[0])
                axes[0].axvline(np.mean(model.ys[:, 0]), color='red')
                axes[1].hist(model.ys[:, 0], color='red')
                axes[1].axvline(np.mean(model.ys[:, 0]), color='red')
                axes[2].hist(ppc[1])
                axes[2].axvline(np.mean(model.ys[:, 1]), color='red')
                axes[3].hist(model.ys[:, 1], color='red')
                axes[3].axvline(np.mean(model.ys[:, 1]), color='red')

        else:
            _, axes = plt.subplots(2, 1, sharex=True)
            axes[0].hist(ppc)
            axes[0].axvline(np.mean(model.ys), color='red')
            axes[1].hist(model.ys, color='red')
            axes[1].axvline(np.mean(model.ys), color='red')
            if log_plot:
                min_val = np.min([np.min(ppc), np.min(model.ys)])
                max_val = np.max([np.max(ppc), np.max(model.ys)])
                logbins = np.geomspace(min_val, max_val, 8)
                axes[0].hist(ppc, bins=logbins)
                axes[0].axvline(np.mean(model.ys), color='red')
                axes[1].hist(model.ys, color='red', bins=logbins)
                axes[1].axvline(np.mean(model.ys), color='red')
                axes[1].set_xscale('log')
                axes[0].set_xscale('log')
            else:
                axes[0].hist(ppc)
                axes[0].axvline(np.mean(model.ys), color='red')
                axes[1].hist(model.ys, color='red')
                axes[1].axvline(np.mean(model.ys), color='red')

        plt.show()
