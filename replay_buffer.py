import numpy as np
import random


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, frame_shape, sample_depth):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.sample_depth = sample_depth
        self.frame_width = frame_shape[0]
        self.frame_height = frame_shape[1]
        self.count = 0
        self.current = 0

        self.rewards = np.empty(self.buffer_size, dtype=np.float32)
        self.actions = np.empty(self.buffer_size, dtype=np.int32)
        self.terminal_flags = np.empty(self.buffer_size, dtype=np.bool)
        self.frames = np.empty((self.buffer_size, self.frame_height, self.frame_width), dtype=np.uint8)

        # for states in a current batch, not all states
        self.states = np.empty((self.batch_size, self.sample_depth, self.frame_height, self.frame_width),
                               dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.sample_depth, self.frame_height, self.frame_width),
                                   dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, done):
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')

        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = done

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.buffer_size

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.sample_depth, self.count - 1)
                if index < self.sample_depth:
                    continue
                if index >= self.current >= index - self.sample_depth:
                    continue
                if self.terminal_flags[index - self.sample_depth:index].any():
                    continue
                break
            self.indices[i] = index

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.sample_depth - 1:
            raise ValueError("Index must be min {}".format(self.sample_depth - 1))
        return self.frames[index - self.sample_depth + 1:index + 1, ...]

    def sample(self):
        if self.count < self.sample_depth:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)

        # transpose frames: (N, D, H, W)->(N, H, W, D)
        # where N - batch size, D - sample depth, H - sample height, W - sample width
        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[self.indices], \
               np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]
