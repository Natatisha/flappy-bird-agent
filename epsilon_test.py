import numpy as np
import math
import matplotlib.pyplot as plt
from utils import plot_rewards, save_rewards, load_rewards, smooth


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
    rewards_0 = load_rewards(file_name='ddqn_rewards_0.npy')
    rewards_1 = load_rewards(file_name='ddqn_rewards.npy')
    y0 = smooth(rewards_0)
    y1 = smooth(rewards_1)

    plt.plot(y0, label='orig')
    plt.plot(y1, label='changed')
    plt.legend()
    plt.show()