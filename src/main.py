import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from LSH import *

mpl.use('tkagg')

def test(data):
    k = [2, 4, 8, 12, 16, 20, 24, 28, 32]
    allmAPs = {}

    allmAPs['Fly'] = {}
    allmAPs['LSH'] = {}


    for i in k:
        flymAP = []
        LSHmAP = []
        
        for _ in range(5):
            FlyModel = FlyLSH(data, data.shape[1], i)
            LSHModel = LSH(data, data.shape[1], i)

            flymAP.append(FlyModel.findmAP())
            LSHmAP.append(LSHModel.findmAP())

        allmAPs['Fly'][i] = (np.mean(flymAP), np.std(flymAP))
        allmAPs['LSH'][i] = (np.mean(LSHmAP), np.std(LSHmAP))
        print(f'{i} done')
    
    return allmAPs


if __name__=='__main__':
    i = 0
    fig, ax = plt.subplots()

    sift = np.genfromtxt("./data/sift.csv", delimiter=',')
    glove = np.genfromtxt("./data/glove.csv", delimiter=',')
    mnist = np.genfromtxt("./data/mnist.csv", delimiter=',')
    gist = np.genfromtxt("./data/gist.csv", delimiter=',')

    dictionary = test(sift)

    for key in dictionary:
        i += 1

        keys = [float(i) for i in list(dictionary[key].keys())]
        values = [tuples[0] for tuples in dictionary[key].values()]
        errors = [tuples[1] for tuples in dictionary[key].values()]

        ax.bar([item - 0.5 * i for item in keys], values, width = 0.5, yerr=errors, label=key)

    plt.ylim([0, 100])
    plt.xlabel("Hash Length(k)")
    plt.ylabel("Mean Average Precision (%)")
    plt.legend()
    plt.show()