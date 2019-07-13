import numpy as np
import math
import matplotlib.pyplot as plt
from utils import plot_rewards, save_rewards, load_rewards


def decay(x, decay_rate=0.99, min_value=0.1):
    eps = decay_rate ** x
    return [max(e, min_value) for e in eps]


def linear_decay(x, decay_rate, min_value=0.01):
    eps = 1. - x * decay_rate
    return [max(e, min_value) for e in eps]


if __name__ == '__main__':
    X = np.arange(0., 70000., 1.)
    # plt.plot(X, linear_decay(X, decay_rate=(1. - 0.01)/30000))
    # plt.show()
    rewards = load_rewards()
    plot_rewards(rewards)