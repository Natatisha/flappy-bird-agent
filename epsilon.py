import numpy as np
import math
import matplotlib.pyplot as plt
from enum import Enum


class EpsilonDecay(Enum):
    LINEAR = 0
    SINUSOID = 1


EPSILON_INITIAL = 1.
EPSILON_CHECKPOINT = 0.1
EPSILON_FINAL = 0.001


class EpsilonGreedyScheduler:
    def __init__(self, max_frames, epsilon_annealing_frames, decay_type=EpsilonDecay.LINEAR,
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
        else:
            return self.linear_decay(frame)

    def linear_decay(self, frame):
        if frame < self.epsilon_annealing_frames:
            return self.slope * frame + self.intercept
        else:
            return max(self.slope_2 * frame + self.intercept_2, self.final)

    def sinusoid_decay(self, x, decay_rate=0.9999998, n_epochs=5):
        return max(
            self.initial * decay_rate ** x * 0.5 * (1. + math.cos((2. * math.pi * x * n_epochs) / self.max_frames)),
            self.final)


# plot epsilon decay
if __name__ == '__main__':
    scheduler = EpsilonGreedyScheduler(600000, 500000, EpsilonDecay.LINEAR)
    X = np.arange(0., 600000, 1.)
    plt.plot(X, [scheduler.get_epsilon(x) for x in X])
    plt.title("Epsilon decay")
    plt.xlabel("Epsilon")
    plt.ylabel("Frames")
    plt.show()
