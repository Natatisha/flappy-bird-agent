import numpy as np
import tensorflow as tf

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport

from replay_buffer import SimpleBuffer
from environment import FlappyBirdWrapper
from epsilon import EpsilonGreedyScheduler, EpsilonDecay
from utils import generate_gif

MAX_EXPERIENCES = 1000000
MIN_EXPERIENCES = 50000

ACTIONS_NUM = 2
EPSILON_DECAY_TYPE = EpsilonDecay.LINEAR
EPSILON_ANNEALING_FRAMES = 2000000
MAX_FRAMES = 30000000
EVAL_FREQUENCY = 200000
EVAL_STEPS = 1000
TARGET_UPD_PERIOD = 50

LEARNING_RATE = 1e-2

SAVE_MODEL_PATH = "dqn_outputs/"
Path(SAVE_MODEL_PATH).mkdir(exist_ok=True)


# a version of HiddenLayer that keeps track of params
class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
        self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
        self.params = [self.W]
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)
        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)


class DQN:
    def __init__(self, D, K, hidden_layer_sizes, gamma, buffer, learning_rate=LEARNING_RATE):
        self.K = K

        # create the graph
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # final layer
        layer = HiddenLayer(M1, K, lambda x: x)
        self.layers.append(layer)

        # collect params for copy
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

        # calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z
        self.predict_op = Y_hat

        selected_action_values = tf.reduce_sum(
            Y_hat * tf.one_hot(self.actions, K),
            reduction_indices=[1]
        )

        cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        # self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)
        # self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(cost)
        # self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

        self.buffer = buffer
        self.gamma = gamma

    def set_session(self, session):
        self.session = session

    def copy_from(self, other):
        # collect all the ops
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        # now run them all
        self.session.run(ops)

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def train(self, target_network):
        states, actions, rewards, dones, next_states = self.buffer.sample()
        next_Q = np.max(target_network.predict(next_states), axis=1)
        targets = [r + self.gamma * next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

        # call optimizer
        self.session.run(
            self.train_op,
            feed_dict={
                self.X: states,
                self.G: targets,
                self.actions: actions
            }
        )

    def add_experience(self, s, a, r, s2, done):
        self.buffer.add_experience(s, a, r, done, s2)

    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            X = np.atleast_2d(x)
            return np.argmax(self.predict(X)[0])


def play_one(env, model, target_model, epsilon_scheduler, total_t, target_upd_period=TARGET_UPD_PERIOD):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done:
        eps = epsilon_scheduler.get_epsilon(total_t)
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        totalreward += reward

        # update the model
        model.add_experience(prev_observation, action, reward, observation, done)
        model.train(target_model)

        iters += 1
        total_t += 1

        if iters % target_upd_period == 0:
            target_model.copy_from(model)

    return totalreward, total_t, iters, eps


def populate_experience(env, replay_buffer):
    print("Populating experience replay buffer...")
    obs = env.reset()
    for i in range(MIN_EXPERIENCES):
        action = np.random.choice(ACTIONS_NUM)
        new_obs, reward, done, info = env.step(action)
        replay_buffer.add_experience(obs, action, reward, done, new_obs)
        obs = new_obs
        if done:
            obs = env.reset()
    env.close()


def train_dqn(gamma, batch_size):
    flappy = FlappyBirdWrapper(screen_output=False)

    D = flappy.observations_num
    K = ACTIONS_NUM

    buffer = SimpleBuffer(MAX_EXPERIENCES, batch_size)

    sizes = [200, 200]
    model = DQN(D, K, sizes, gamma, buffer)
    target_model = DQN(D, K, sizes, gamma, buffer)
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        model.set_session(session)
        target_model.set_session(session)

        epsilon_scheduler = EpsilonGreedyScheduler(decay_type=EpsilonDecay.LINEAR, max_frames=MAX_FRAMES,
                                                   epsilon_annealing_frames=EPSILON_ANNEALING_FRAMES)
        populate_experience(flappy, buffer)

        rewards = []
        total_t = 0
        while total_t < MAX_FRAMES:
            epoch_frame = 0
            while epoch_frame < EVAL_FREQUENCY:
                episode_reward, total_t, iters, epsilon = play_one(flappy, model, target_model, epsilon_scheduler,
                                                                   total_t)
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

                # flappy.render()
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
