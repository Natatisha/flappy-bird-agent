import numpy as np
import matplotlib.pyplot as plt


def smooth(x):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


def plot_rewards(episode_rewards):
    y = smooth(episode_rewards)
    plt.plot(episode_rewards, label='orig')
    plt.plot(y, label='smoothed')
    plt.legend()
    plt.show()


def save_rewards(episode_rewards, file_name='ddqn_rewards.npy'):
    np.save(file_name, episode_rewards)


def load_rewards(file_name='ddqn_rewards.npy'):
    return np.load(file_name)
