import numpy as np

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport

from utils import generate_gif, save_rewards
from environment import FlappyBirdWrapper
from epsilon import EpsilonDecay, EpsilonGreedyScheduler

ACTIONS_NUM = 2
MAX_FRAMES = 3000000
EPSILON_ANNEALING_FRAMES = 2500000
EVAL_FREQUENCY = 100000
EVAL_STEPS = 1000
LEARNING_RATE = 0.01

SAVE_MODEL_PATH = "q_learning_outputs/"

Path(SAVE_MODEL_PATH).mkdir(exist_ok=True)


def create_state_grid(low, high, bins=(10, 10, 10, 10)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    return grid


def discretize(sample, grid):
    return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension


class QLearningAgent:

    def __init__(self, env, alpha=LEARNING_RATE, gamma=0.99):
        self.env = env
        self.state_grid = create_state_grid(env.low_obs_values, env.high_obs_values, bins=(10, 10, 20, 10))
        self.state_sizes = tuple(len(splits) + 1 for splits in self.state_grid)
        self.action_size = ACTIONS_NUM

        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.q_table = np.zeros(shape=(self.state_sizes + (self.action_size,)))

    def preprocess_state(self, state):
        return tuple(discretize(state, self.state_grid))

    def predict(self, state):
        state = self.preprocess_state(state)
        return self.q_table[state]

    def learn(self, state, action, reward, next_state, done):
        value = reward + self.gamma * max(self.predict(next_state)) * (1 - done)
        state = self.preprocess_state(state)
        self.q_table[state + (action,)] = self.alpha * value - self.q_table[state + (action,)]

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return np.random.choice(self.action_size)
        else:
            p = self.predict(s)
            return np.argmax(p)

    def save(self, file_name=SAVE_MODEL_PATH + 'q_learning_weights.npy'):
        np.save(file_name, self.q_table)

    def load(self, file_name=SAVE_MODEL_PATH + 'q_learning_weights.npy'):
        self.q_table = np.load(file_name)
        return self.q_table


def play_one(env, model, epsilon_scheduler, total_t):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done:
        eps = epsilon_scheduler.get_epsilon(total_t)
        action = model.sample_action(observation, 1.)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        totalreward += reward

        model.learn(prev_observation, action, reward, observation, done)

        iters += 1
        total_t += 1

    return totalreward, total_t, iters, eps


def train_q_learning_model(gamma):
    flappy = FlappyBirdWrapper(screen_output=False)
    model = QLearningAgent(flappy, LEARNING_RATE, gamma)
    epsilon_scheduler = EpsilonGreedyScheduler(decay_type=EpsilonDecay.LINEAR, max_frames=MAX_FRAMES,
                                               epsilon_annealing_frames=EPSILON_ANNEALING_FRAMES)
    rewards = []
    total_t = 0
    while total_t < MAX_FRAMES:
        epoch_frame = 0
        while epoch_frame < EVAL_FREQUENCY:
            episode_reward, total_t, iters, epsilon = play_one(flappy, model, epsilon_scheduler, total_t)
            epoch_frame += iters
            rewards.append(episode_reward)
            if len(rewards) % 100 == 0:
                print("Episode:", len(rewards),
                      "Frame number:", total_t,
                      "Episode reward:", episode_reward,
                      "Avg Reward (Last 100):", "%.3f" % np.mean(rewards[-100:]),
                      "Epsilon:", "%.3f" % epsilon)

        # Evaluate
        done = True
        gif = True
        frames_for_gif = []
        eval_rewards = []
        evaluate_frame_number = 0

        for _ in range(EVAL_STEPS):
            if done:
                state = flappy.reset()
                episode_reward_sum = 0
                done = False

            action = model.sample_action(state, 0.)

            state, reward, done, new_frame = flappy.step(action, False)
            evaluate_frame_number += 1
            episode_reward_sum += reward

            if gif:
                frames_for_gif.append(flappy.get_screen())
            if done:
                eval_rewards.append(episode_reward_sum)
                gif = False  # Save only the first game of the evaluation as a gif

        eval_score = np.mean(eval_rewards)
        print("Evaluation score:\n", eval_score)
        try:
            generate_gif(frames_for_gif, total_t, eval_score, SAVE_MODEL_PATH)
        except IndexError:
            print("No evaluation game finished")

    model.save()
    save_rewards(rewards, file_name='q_learning_rewards.npy')
