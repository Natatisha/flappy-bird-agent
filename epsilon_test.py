import numpy as np
import math
import matplotlib.pyplot as plt
from utils import plot_rewards, save_rewards, load_rewards, smooth
from ddqn import EpsilonGreedyScheduler, EpsilonDecay


def decay(x, decay_rate=0.999, min_value=0.1):
    eps = decay_rate ** x
    return max(eps, min_value)


def linear_decay(x, initial_eps, decay_rate, min_value=0.01):
    eps = initial_eps - x * decay_rate
    return max(eps, min_value)


def sinusoid_decay(x, initial_eps, X_total, decay_rate=0.9996, n_epochs=5):
    return initial_eps * decay_rate ** x * 0.5 * (1. + math.cos((2. * math.pi * x * n_epochs) / X_total))


if __name__ == '__main__':
    scheduler = EpsilonGreedyScheduler(EpsilonDecay.SINUSOID)
    X = np.arange(0., 3e6, 1.)
    plt.plot(X, [scheduler.get_epsilon(x) for x in X])
    # plt.plot(X, [decay(x, 0.9997) for x in X])
    plt.show()
    rewards_0 = load_rewards(file_name='ddqn_rewards_0.npy')
    rewards_1 = load_rewards(file_name='ddpq_weights_20000.npy')
    y0 = smooth(rewards_0)
    y1 = smooth(rewards_1)

    # plt.plot(y0, label='orig')
    # plt.plot(y1, label='changed')
    # plt.legend()
    # plt.show()

    # plot_rewards(rewards_1[:10000])
