import numpy as np
import math
import matplotlib.pyplot as plt
from utils import plot_rewards, save_rewards, load_rewards, smooth
from enum import Enum

class EpsilonDecay(Enum):
    LINEAR = 0
    SINUSOID = 1
    EXPONENTIAL = 2


EPSILON_INITIAL = 1.
EPSILON_CHECKPOINT = 0.1
EPSILON_FINAL = 0.001


class EpsilonGreedyScheduler:
    def __init__(self, max_frames, epsilon_annealing_frames, decay_type,
                 epsilon_initial_value=EPSILON_INITIAL,
                 epsilon_checkpoint=EPSILON_CHECKPOINT, epsilon_final_value=EPSILON_FINAL):
        self.decay_type = decay_type
        self.initial = epsilon_initial_value
        self.final = epsilon_final_value
        self.chekpoint = epsilon_checkpoint
        self.epsilon_annealing_frames = epsilon_annealing_frames
        self.max_frames = max_frames

        self.slope = -(self.initial - self.chekpoint) / epsilon_annealing_frames
        self.intercept = self.initial - self.slope
        self.slope_2 = -(self.chekpoint - self.final) / (max_frames - epsilon_annealing_frames)
        self.intercept_2 = self.final - self.slope_2 * max_frames

    def get_epsilon(self, frame):
        if self.decay_type == EpsilonDecay.SINUSOID:
            return self.sinusoid_decay(frame)
        elif self.decay_type == EpsilonDecay.EXPONENTIAL:
            return self.exp_decay(frame)
        else:
            return self.linear_decay(frame)

    def exp_decay(self, frame, decay_rate=0.999999):
        eps = decay_rate ** frame
        return max(eps, self.final)

    def linear_decay(self, frame):
        if frame < self.epsilon_annealing_frames:
            return self.slope * frame + self.intercept
        else:
            return max(self.slope_2 * frame + self.intercept_2, self.final)

    def sinusoid_decay(self, x, decay_rate=0.999997, n_epochs=5):
        return max(
            self.initial * decay_rate ** x * 0.5 * (1. + math.cos((2. * math.pi * x * n_epochs) / self.max_frames)),
            self.final)


if __name__ == '__main__':
    scheduler = EpsilonGreedyScheduler(EpsilonDecay.SINUSOID, )
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