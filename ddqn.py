import sys
import os

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport

import tensorflow as tf
import numpy as np

from epsilon import EpsilonGreedyScheduler, EpsilonDecay
from environment import FlappyBirdWrapper
from image_transformer import ImageTransformer
from replay_buffer import ReplayBuffer

from utils import generate_gif

# constants
# for testing
# MAX_EXPERIENCES = 10000
# MIN_EXPERIENCES = 100

MAX_EXPERIENCES = 1000000
MIN_EXPERIENCES = 50000

ACTIONS_NUM = 2
EPSILON_DECAY_TYPE = EpsilonDecay.LINEAR
EPSILON_ANNEALING_FRAMES = 2000000
MAX_FRAMES = 30000000
OBS_SHAPE = (512, 288, 3)
CROP_BOUNDS = (0, 50, 400, 238)

HIDDEN_LAYER_SIZE = 512
TARGET_UPD_PERIOD = 10000
IMG_SIZE = 84
LEARNING_RATE = 1e-6
FRAMES_IN_STATE = 4

SAVE_MODEL_PATH = "outputs/"
SUMMARIES = "summaries/"
RUNID = 'run_1'

Path(SAVE_MODEL_PATH).mkdir(exist_ok=True)
Path(SUMMARIES + RUNID).mkdir(parents=True, exist_ok=True)
SUMM_WRITER = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID))

MAX_EPISODE_LENGTH = 18000  # Equivalent of 5 minutes of gameplay at 60 frames per second
EVAL_FREQUENCY = 500000  # Number of frames the agent sees between evaluations
EVAL_STEPS = 10000  # Number of frames for one evaluation

UPDATE_FREQ = 4  # Every four actions a gradient descend step is performed


class DDQN:

    def __init__(self, actions_n, hidden_layers_size, scope, learning_rate=1e-6,
                 frame_shape=(IMG_SIZE, IMG_SIZE),
                 agent_history_length=FRAMES_IN_STATE):
        self.actions_n = actions_n
        self.scope = scope
        self.lr = learning_rate

        with tf.variable_scope(self.scope):
            self.X = tf.placeholder(dtype=tf.float32,
                                    shape=(None, frame_shape[0], frame_shape[1], agent_history_length),
                                    name='X')
            # self.X_scaled = self.X / 255

            initializer = tf.variance_scaling_initializer(scale=2)
            padding = "VALID"

            self.conv1 = tf.layers.conv2d(self.X, filters=32, kernel_size=[8, 8], strides=4,
                                          kernel_initializer=initializer, padding=padding,
                                          activation=tf.nn.relu, use_bias=False, name='conv1')
            self.conv2 = tf.layers.conv2d(self.conv1, filters=64, kernel_size=[4, 4], strides=2,
                                          kernel_initializer=initializer, padding=padding,
                                          activation=tf.nn.relu, use_bias=False, name='conv2')
            self.conv3 = tf.layers.conv2d(self.conv2, filters=64, kernel_size=[3, 3], strides=1,
                                          kernel_initializer=initializer, padding=padding,
                                          activation=tf.nn.relu, use_bias=False, name='conv3')
            self.conv4 = tf.layers.conv2d(self.conv3, filters=hidden_layers_size, kernel_size=[7, 7], strides=1,
                                          kernel_initializer=initializer, padding=padding,
                                          activation=tf.nn.relu, use_bias=False, name='conv4')

            self.advantagestream, self.valuestream = tf.split(self.conv4, 2, 3)

            self.advantagestream = tf.contrib.layers.flatten(self.advantagestream)
            self.valuestream = tf.contrib.layers.flatten(self.valuestream)

            self.advantage = tf.contrib.layers.fully_connected(self.advantagestream, self.actions_n,
                                                               weights_initializer=tf.variance_scaling_initializer(
                                                                   scale=2),
                                                               scope='advantage')
            self.value = tf.contrib.layers.fully_connected(self.valuestream, 1,
                                                           weights_initializer=tf.variance_scaling_initializer(scale=2),
                                                           scope='value')

            self.q_values = tf.add(self.value,
                                   tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True)),
                                   name='q_values')

            self.actions = tf.placeholder(dtype=tf.int32, shape=(None,), name='actions')
            self.target_q = tf.placeholder(dtype=tf.float32, shape=(None,), name='target_q')

            self.best_action = tf.argmax(self.q_values, axis=1)
            self.Q = tf.reduce_sum(self.q_values * tf.one_hot(self.actions, actions_n), axis=1)
            self.cost = tf.reduce_mean(tf.losses.huber_loss(labels=self.target_q, predictions=self.Q))
            self.train_optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

            self.trainable_vars = self.collect_trainable_vars()

            self.sess = None
            self.saver = None

    def collect_trainable_vars(self):
        collection = tf.GraphKeys.TRAINABLE_VARIABLES
        variables = tf.get_collection(collection, scope=self.scope)
        assert len(variables) > 0
        print("Variables in scope '{}':".format(self.scope))
        for v in variables:
            print("\t" + str(v))
        return variables

    def copy_from(self, other):
        self.sess.run([v_t.assign(v) for v_t, v in zip(other.trainable_vars, self.trainable_vars)])

    def set_session(self, session):
        self.sess = session

    def predict(self, X):
        return self.sess.run(self.q_values, feed_dict={self.X: X})

    def get_best_action(self, X):
        return self.sess.run(self.best_action, feed_dict={self.X: X})

    def train(self, states, actions, targets):
        cost, _ = self.sess.run([self.cost, self.train_optimizer], feed_dict={
            self.actions: actions,
            self.target_q: targets,
            self.X: states
        })
        return cost

    def sample_action(self, state, eps):
        if np.random.random() < eps:
            return np.random.choice(self.actions_n, p=[0.6, 0.4])  # better not to act than act
            # return np.random.choice(self.actions_n)
        else:
            return self.get_best_action([state])[0]

    def save(self, frame_number, path=SAVE_MODEL_PATH, write_meta_graph=True):
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=4)
        self.saver.save(self.sess, path + 'my_model', global_step=frame_number, write_meta_graph=write_meta_graph)


def update_state(state, new_frame):
    return np.append(state[:, :, 1:], np.expand_dims(new_frame, 2), axis=2)


def learn(model, target_model, replay_buffer, gamma):
    states, actions, rewards, next_states, dones = replay_buffer.sample()

    best_actions = model.get_best_action(next_states)
    next_Qs = target_model.predict(next_states)
    next_Q = next_Qs[range(replay_buffer.batch_size), best_actions]

    # if terminal state - target is reward, else r + gamma*max(next_Q)
    targets = rewards + gamma * next_Q * np.invert(dones).astype(np.float32)

    loss = model.train(states, actions, targets)
    return loss


def populate_experience(env, replay_buffer):
    print("Populating experience replay buffer...")
    obs = env.reset()
    for i in range(MIN_EXPERIENCES):
        action = np.random.choice(ACTIONS_NUM)
        processed_new_frame, reward, done, new_frame = env.step(action)
        replay_buffer.add_experience(action, processed_new_frame, reward, done)
        if done:
            obs = env.reset()
    env.close()


def train_ddqn_model(batch_size, gamma):
    tf.reset_default_graph()

    replay_buffer = ReplayBuffer(MAX_EXPERIENCES, batch_size, (IMG_SIZE, IMG_SIZE), FRAMES_IN_STATE)
    episode_rewards = []

    model = DDQN(
        ACTIONS_NUM,
        HIDDEN_LAYER_SIZE,
        learning_rate=LEARNING_RATE,
        scope="model")

    target_model = DDQN(
        ACTIONS_NUM,
        HIDDEN_LAYER_SIZE,
        learning_rate=LEARNING_RATE,
        scope="target_model")

    with tf.name_scope('Performance'):
        LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
        LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)
        REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
        REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)
        EVAL_SCORE_PH = tf.placeholder(tf.float32, shape=None, name='evaluation_summary')
        EVAL_SCORE_SUMMARY = tf.summary.scalar('evaluation_score', EVAL_SCORE_PH)

    PERFORMANCE_SUMMARIES = tf.summary.merge([LOSS_SUMMARY, REWARD_SUMMARY])

    image_transformer = ImageTransformer(origin_shape=OBS_SHAPE, out_shape=(IMG_SIZE, IMG_SIZE),
                                         crop_boundaries=CROP_BOUNDS)
    epsilon_scheduler = EpsilonGreedyScheduler(decay_type=EpsilonDecay.LINEAR, max_frames=MAX_FRAMES,
                                               epsilon_annealing_frames=EPSILON_ANNEALING_FRAMES)
    flappy = FlappyBirdWrapper(image_transformer=image_transformer)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        frame_number = 0
        rewards = []
        loss_list = []

        model.set_session(sess)
        target_model.set_session(sess)
        flappy.set_session(sess)
        sess.run(init)

        populate_experience(flappy, replay_buffer)

        while frame_number < MAX_FRAMES:
            epoch_frame = 0
            while epoch_frame < EVAL_FREQUENCY:
                state = flappy.reset()
                episode_reward_sum = 0
                for _ in range(MAX_EPISODE_LENGTH):
                    epsilon = epsilon_scheduler.get_epsilon(frame_number)
                    action = model.sample_action(state, epsilon)
                    processed_new_frame, reward, done, _ = flappy.step(action)
                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward

                    new_state = update_state(state, processed_new_frame)
                    state = new_state

                    replay_buffer.add_experience(action, processed_new_frame, reward, done)

                    if frame_number % UPDATE_FREQ == 0:
                        loss = learn(model, target_model, replay_buffer, gamma)
                        loss_list.append(loss)
                    if frame_number % TARGET_UPD_PERIOD == 0:
                        target_model.copy_from(model)
                    if done:
                        break

                rewards.append(episode_reward_sum)

                # Output the progress:
                if len(rewards) % 10 == 0:
                    summ = sess.run(PERFORMANCE_SUMMARIES,
                                    feed_dict={LOSS_PH: np.mean(loss_list),
                                               REWARD_PH: np.mean(rewards[-100:])})

                    SUMM_WRITER.add_summary(summ, frame_number)
                    loss_list = []
                    print("Episode:", len(rewards),
                          "Frame number:", frame_number,
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

                processed_new_frame, reward, done, new_frame = flappy.step(action)
                evaluate_frame_number += 1
                episode_reward_sum += reward
                new_state = update_state(state, processed_new_frame)
                state = new_state
                
                if gif:
                    frames_for_gif.append(new_frame)
                if done:
                    eval_rewards.append(episode_reward_sum)
                    gif = False  # Save only the first game of the evaluation as a gif

            eval_score = np.mean(eval_rewards)
            print("Evaluation score:\n", eval_score)
            try:
                generate_gif(frames_for_gif, frame_number, eval_score, SAVE_MODEL_PATH)
            except IndexError:
                print("No evaluation game finished")

            # Show the evaluation score in tensorboard
            summ = sess.run(EVAL_SCORE_SUMMARY, feed_dict={EVAL_SCORE_PH: np.mean(eval_rewards)})
            SUMM_WRITER.add_summary(summ, frame_number)
