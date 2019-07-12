import numpy as np
import math
import matplotlib.pyplot as plt


def decay(x, decay_rate=0.99, min_value=0.1):
    eps = decay_rate ** x
    return [max(e, min_value) for e in eps]


def linear_decay(x, decay_rate, min_value=0.01):
    eps = 1. - x * decay_rate
    return [max(e, min_value) for e in eps]


if __name__ == '__main__':
    X = np.arange(0., 30000., 1.)
    plt.plot(X, linear_decay(X, decay_rate=(1. - 0.01)/50000))
    plt.show()
