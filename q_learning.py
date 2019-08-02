import numpy as np
import pandas as pd
import gym

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport

from utils import generate_gif, save_rewards
from environment import FlappyBirdWrapper
from epsilon import EpsilonDecay, EpsilonGreedyScheduler

ACTIONS_NUM = 2
MAX_FRAMES = 10
EPSILON_ANNEALING_FRAMES = 3000
EVAL_FREQUENCY = 10
EVAL_STEPS = 10
LEARNING_RATE = 0.001

SAVE_MODEL_PATH = "q_learning_outputs/"

Path(SAVE_MODEL_PATH).mkdir(exist_ok=True)


# turns list of integers into an int
# Ex.
# build_state([1,2,3,4,5]) -> 12345
def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))


def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


class FeatureTransformer:
    def __init__(self):
        self.agent_pos_y = np.linspace(-2.4, 2.4, 9)
        self.agent_velocity = np.linspace(-2, 2, 9)
        self.dist_next = np.linspace(-0.4, 0.4, 9)
        self.dist_next_next = np.linspace(-3.5, 3.5, 9)

    def transform(self, observation):
        # returns an int
        agent_pos, agent_vel, dist_next, dist_next_next = observation
        return build_state([
            to_bin(agent_pos, self.agent_pos_y),
            to_bin(agent_vel, self.agent_velocity),
            to_bin(dist_next, self.dist_next),
            to_bin(dist_next_next, self.dist_next_next),
        ])


class QLearningBinsModel:
    def __init__(self, states_n, actions_n, feature_transformer, learning_rate=LEARNING_RATE, gamma=0.99):
        self.actions_n = actions_n
        self.feature_transformer = feature_transformer
        self.lr = learning_rate
        self.gamma = gamma

        num_states = 10 ** states_n
        self.Q = np.random.uniform(low=-1, high=1, size=(num_states, self.actions_n))

    def predict(self, s):
        x = self.feature_transformer.transform(s)
        return self.Q[x]

    def learn(self, state, action, reward, next_state, done):
        x = self.feature_transformer.transform(state)
        G = reward + self.gamma * np.max(self.predict(next_state)) * (1 - done)
        self.Q[x, action] += self.lr * (G - self.Q[x, action])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return np.random.choice(self.actions_n)
        else:
            p = self.predict(s)
            return np.argmax(p)

    def save(self):
        pass


class QLearningModel:
    def __init__(self, actions_num=ACTIONS_NUM, learning_rate=LEARNING_RATE, gamma=0.99):
        self.actions_n = actions_num
        self.lr = learning_rate
        self.gamma = gamma
        # self.q_table = pd.DataFrame(columns=list(range(self.actions_n)), dtype=np.float32)
        self.q_table = dict()

    def sample_action(self, state, epsilon):
        state = tuple(state)
        self.check_state_exist(state)
        # if np.random.uniform() < epsilon:
        #     state_actions = self.q_table[state]
        #     action = np.argmax(state_actions)
        # else:
        action = np.random.choice(self.actions_n)
        return action

    def learn(self, state, action, reward, next_state, done):
        state = tuple(state)
        next_state = tuple(next_state)
        self.check_state_exist(next_state)
        q_predict = self.q_table[state][action]
        q_target = reward + self.gamma * self.q_table[next_state][action] * (1 - done)
        # print("Q_TARGET {}".format(q_target))
        self.q_table[state][action] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table:
            # self.q_table = self.q_table.append(
            #     pd.Series(
            #         np.zeros(self.actions_n),
            #         index=self.q_table.columns,
            #         name=state))
            self.q_table[state] = np.zeros(self.actions_n)

    def save(self, file_path=SAVE_MODEL_PATH + 'q_learning_model'):
        # self.q_table.to_csv(file_path)
        pass


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

    return totalreward, total_t, eps


def train_q_learning_model(gamma):
    flappy = FlappyBirdWrapper(screen_output=False)
    feature_transformer = FeatureTransformer()

    # model = QLearningBinsModel(feature_transformer=feature_transformer, states_n=flappy.observations_num,
    #                            actions_n=ACTIONS_NUM, gamma=gamma)
    model = QLearningModel(gamma=gamma)
    # flappy = gym.make("MountainCar-v0")
    epsilon_scheduler = EpsilonGreedyScheduler(decay_type=EpsilonDecay.LINEAR, max_frames=MAX_FRAMES,
                                               epsilon_annealing_frames=EPSILON_ANNEALING_FRAMES)
    rewards = []
    states = []
    total_t = 0
    for i_episode in range(MAX_FRAMES):
        episode_reward, total_t, epsilon = play_one(flappy, model, epsilon_scheduler, total_t)
        rewards.append(episode_reward)
        # if len(rewards) % 100 == 0:
        print("Episode:", len(rewards),
              "Frame number:", total_t,
              "Episode reward:", episode_reward,
              "Avg Reward (Last 100):", "%.3f" % np.mean(rewards[-100:]),
              "Epsilon:", "%.3f" % epsilon)
        # if len(rewards) > 2:
        #     return

        # Evaluate
        done = True
        gif = True
        frames_for_gif = []
        eval_rewards = []
        evaluate_frame_number = 0

        # for _ in range(EVAL_STEPS):
        #     if done:
        #         state = flappy.reset()
        #         episode_reward_sum = 0
        #         done = False
        #
        #     # flappy.render()
        #     action = model.sample_action(state, 0.)
        #
        #     processed_new_frame, reward, done, new_frame = flappy.step(action)
        #     evaluate_frame_number += 1
        #     episode_reward_sum += reward
        #
        #     if gif:
        #         frames_for_gif.append(new_frame)
        #     if done:
        #         eval_rewards.append(episode_reward_sum)
        #         gif = False  # Save only the first game of the evaluation as a gif

        # eval_score = np.mean(eval_rewards)
        # print("Evaluation score:\n", eval_score)
        # try:
        #     generate_gif(frames_for_gif, frame_number, eval_score, SAVE_MODEL_PATH)
        # except IndexError:
        #     print("No evaluation game finished")

        # print(model.q_table.values())
    # print(len(model.q_table.values()))
    model.save()
    save_rewards(rewards, file_name='q_learning_rewards.npy')
