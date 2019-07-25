import sys
import os

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport

import tensorflow as tf
import numpy as np
from datetime import datetime
from enum import Enum
import math

from image_transformer import ImageTransformer
from replay_buffer import ReplayBuffer

from utils import generate_gif


class EpsilonDecay(Enum):
    LINEAR = 0
    SINUSOID = 1
    EXPONENTIAL = 2


# constants
# for testing
# MAX_EXPERIENCES = 10000
# MIN_EXPERIENCES = 100

# prod
# Flappy
# EPSILON_DECAY_TYPE = EpsilonDecay.LINEAR
# EPSILON_INITIAL = 1.
# EPSILON_CHECKPOINT = 0.1
# EPSILON_FINAL = 0.001
# EPSILON_ANNEALING_FRAMES = 1e6
# MAX_FRAMES = 2e6
# OBS_SHAPE=(512, 288, 3)
# CROP_BOUNDS=(0, 50, 400, 238)

# Pong
# EPSILON_DECAY_TYPE = EpsilonDecay.LINEAR
# EPSILON_INITIAL = 1.
# EPSILON_CHECKPOINT = 0.1
# EPSILON_FINAL = 0.01
# EPSILON_ANNEALING_FRAMES = 1e6
# MAX_FRAMES = 2e6
# OBS_SHAPE = (210, 160, 3)
# CROP_BOUNDS = (33, 0, 160, 160)
# UPDATE_FREQ = 4
# ACTIONS_NUM = 4

# Breakout
EPSILON_DECAY_TYPE = EpsilonDecay.LINEAR
EPSILON_INITIAL = 1.
EPSILON_CHECKPOINT = 0.1
EPSILON_FINAL = 0.01
EPSILON_ANNEALING_FRAMES = 5e5
MAX_FRAMES = 1e6
OBS_SHAPE = (210, 160, 3)
CROP_BOUNDS = (34, 0, 160, 160)
ACTIONS_NUM = 4

MAX_EXPERIENCES = 1000000
MIN_EXPERIENCES = 50000

HIDDEN_LAYER_SIZE = 512
TARGET_UPD_PERIOD = 10000
IMG_SIZE = 80
LEARNING_RATE = 1e-5
FRAMES_IN_STATE = 4
SAVE_REWARD_EACH = 1000
SAVE_MODEL_EACH = 10000

SAVE_MODEL_PATH = "outputs/"
SUMMARIES = "summaries/"
RUNID = 'run_1'

Path(SAVE_MODEL_PATH).mkdir(exist_ok=True)
Path(SUMMARIES + RUNID).mkdir(parents=True, exist_ok=True)
SUMM_WRITER = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID))


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
            self.X_scaled = self.X / 255

            initializer = tf.variance_scaling_initializer(scale=2)
            padding = "VALID"

            self.conv1 = tf.layers.conv2d(self.X_scaled, filters=32, kernel_size=[8, 8], strides=4,
                                          kernel_initializer=initializer, padding=padding,
                                          activation=tf.nn.relu, use_bias=False, name='conv1')
            self.conv2 = tf.layers.conv2d(self.conv1, filters=64, kernel_size=[4, 4], strides=2,
                                          kernel_initializer=initializer, padding=padding,
                                          activation=tf.nn.relu, use_bias=False, name='conv2')
            self.conv3 = tf.layers.conv2d(self.conv2, filters=64, kernel_size=[3, 3], strides=1,
                                          kernel_initializer=initializer, padding=padding,
                                          activation=tf.nn.relu, use_bias=False, name='conv3')
            # self.conv4 = tf.layers.conv2d(self.conv3, filters=hidden_layers_size, kernel_size=[7, 7], strides=1,
            #                               kernel_initializer=initializer, padding=padding,
            #                               activation=tf.nn.relu, use_bias=False, name='conv4')

            # self.advantagestream, self.valuestream = tf.split(self.conv4, 2, 3)

            # self.advantagestream = tf.contrib.layers.flatten(self.advantagestream)
            # self.valuestream = tf.contrib.layers.flatten(self.valuestream)
            #
            # self.advantage = tf.contrib.layers.fully_connected(self.advantagestream, self.actions_n,
            #                                                    weights_initializer=tf.variance_scaling_initializer(
            #                                                        scale=2),
            #                                                    scope='advantage')
            # self.value = tf.contrib.layers.fully_connected(self.valuestream, 1,
            #                                                weights_initializer=tf.variance_scaling_initializer(scale=2),
            #                                                scope='value')
            #
            # self.q_values = tf.add(self.value,
            #                        tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True)),
            #                        name='q_values')
            self.flatten = tf.reshape(self.conv3, [-1, np.prod(self.conv3.shape.as_list()[1:])], name='flatten')
            self.fc1 = tf.layers.dense(self.flatten, hidden_layers_size, activation=tf.nn.relu,
                                       kernel_initializer=initializer, name='fc1')
            self.q_values = tf.layers.dense(self.fc1, ACTIONS_NUM, activation=tf.nn.relu,
                                            kernel_initializer=initializer, name='predicted_actions')

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
            # return np.random.choice(self.actions_n, p=[0.6, 0.4])  # better to act 1 time in 5 steps
            return np.random.choice(self.actions_n)  # better to act 1 time in 5 steps
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


class EpsilonGreedyScheduler:
    def __init__(self, decay_type=EPSILON_DECAY_TYPE, epsilon_initial_value=EPSILON_INITIAL,
                 epsilon_checkpoint=EPSILON_CHECKPOINT, epsilon_final_value=EPSILON_FINAL):
        self.decay_type = decay_type
        self.initial = epsilon_initial_value
        self.final = epsilon_final_value
        self.chekpoint = epsilon_checkpoint

        self.slope = -(self.initial - self.chekpoint) / EPSILON_ANNEALING_FRAMES
        self.intercept = self.initial - self.slope
        self.slope_2 = -(self.chekpoint - self.final) / (MAX_FRAMES - EPSILON_ANNEALING_FRAMES)
        self.intercept_2 = self.final - self.slope_2 * MAX_FRAMES

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
        if frame < EPSILON_ANNEALING_FRAMES:
            return self.slope * frame + self.intercept
        else:
            return max(self.slope_2 * frame + self.intercept_2, self.final)

    def sinusoid_decay(self, x, decay_rate=0.999997, n_epochs=5):
        return max(self.initial * decay_rate ** x * 0.5 * (1. + math.cos((2. * math.pi * x * n_epochs) / MAX_FRAMES)),
                   self.final)


def play_one_episode(
        env,
        session,
        total_t,
        model,
        target_model,
        replay_buffer,
        image_tansformer,
        gamma,
        epsilon_scheduler,
        target_upd_period=TARGET_UPD_PERIOD):
    t0 = datetime.now()
    total_training_time = 0

    raw_frame = env.reset()
    frame = image_tansformer.transform(raw_frame, session)
    state = np.stack([frame] * FRAMES_IN_STATE, axis=2)
    loss = None

    done = False

    episode_reward = 0
    num_steps = 0

    while not done:

        if total_t % target_upd_period == 0:
            target_model.copy_from(model)
            print("Copied model parameters to target network. total_t = %s, period = %s" % (
                total_t, target_upd_period))

        epsilon = epsilon_scheduler.get_epsilon(total_t)
        action = model.sample_action(state, epsilon)
        raw_frame, reward, done, _ = env.step(action)
        frame = image_tansformer.transform(raw_frame, session)
        next_state = update_state(state, frame)
        episode_reward += reward

        replay_buffer.add_experience(action, frame, reward, done)

        t0_2 = datetime.now()
        # if total_t % UPDATE_FREQ == 0:
        loss = learn(model, target_model, replay_buffer, gamma)
        dt = datetime.now() - t0_2

        total_training_time += dt.total_seconds()
        num_steps += 1
        total_t += 1

        state = next_state

    return total_t, episode_reward, loss, (datetime.now() - t0).total_seconds(), \
           num_steps, total_training_time / num_steps, epsilon


def populate_experience(env, image_transformer, replay_buffer, sess):
    print("Populating experience replay buffer...")
    obs = env.reset()
    for i in range(MIN_EXPERIENCES):

        action = np.random.choice(ACTIONS_NUM)
        obs, reward, done, _ = env.step(action)
        obs_small = image_transformer.transform(obs, sess)  # not used anymore
        replay_buffer.add_experience(action, obs_small, reward, done)
        if done:
            env.reset()
    env.close()


def train_ddqn_model(env, num_episodes, batch_size, gamma):
    replay_buffer = ReplayBuffer(MAX_EXPERIENCES, batch_size, (IMG_SIZE, IMG_SIZE), FRAMES_IN_STATE)
    episode_rewards = np.zeros(num_episodes)
    losses = []

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

    image_transformer = ImageTransformer(origin_shape=OBS_SHAPE, out_shape=(IMG_SIZE, IMG_SIZE),
                                         crop_boundaries=CROP_BOUNDS)
    epsilon_scheduler = EpsilonGreedyScheduler(EpsilonDecay.LINEAR)

    total_t = 0

    with tf.name_scope('Performance'):
        LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
        LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)
        REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
        REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)

    PERFORMANCE_SUMMARIES = tf.summary.merge([LOSS_SUMMARY, REWARD_SUMMARY])

    with tf.Session() as sess:
        model.set_session(sess)
        target_model.set_session(sess)
        sess.run(tf.global_variables_initializer())

        populate_experience(env, image_transformer, replay_buffer, sess)

        t0 = datetime.now()
        for i in range(num_episodes):
            total_t, episode_reward, loss, duration, num_steps_in_episode, time_per_step, epsilon = play_one_episode(
                env,
                sess,
                total_t,
                model,
                target_model,
                replay_buffer,
                image_transformer,
                gamma,
                epsilon_scheduler
            )
            episode_rewards[i] = episode_reward

            if loss is not None:
                losses.append(loss)

            last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()

            if total_t % SAVE_REWARD_EACH == 0:
                print("Saving rewards to tensorboard")
                summ = sess.run(PERFORMANCE_SUMMARIES, feed_dict={LOSS_PH: np.mean(losses),
                                                                  REWARD_PH: last_100_avg})
                SUMM_WRITER.add_summary(summ, total_t)
            if total_t % SAVE_MODEL_EACH == 0:
                print("Saving the model")
                model.save(total_t,
                           write_meta_graph=(total_t <= SAVE_MODEL_EACH))  # save meta graph for the first time only

                evaluate_model(env, image_transformer, model, sess, total_t)

            print("Episode:", i,
                  "Duration:", duration,
                  "Num steps:", num_steps_in_episode,
                  "Reward:", episode_reward,
                  "Training time per step:", "%.3f" % time_per_step,
                  "Avg Reward (Last 100):", "%.3f" % last_100_avg,
                  "Epsilon:", "%.3f" % epsilon
                  )
            sys.stdout.flush()
        print("Total duration:", str(datetime.now() - t0))

        model.save(total_t)

    return model, episode_rewards


def evaluate_model(env, image_transformer, model, sess, total_t):
    # Evaluate model
    frames_eval = []
    eval_reward = 0
    obs = env.reset()
    frames_eval.append(obs)
    frame = image_transformer.transform(obs, sess)
    state = np.stack([frame] * FRAMES_IN_STATE, axis=2)
    done = False
    while not done:
        action = model.sample_action(state, 0.)
        next_frame, reward, done, _ = env.step(action)
        frame = image_transformer.transform(next_frame, sess)
        next_state = update_state(state, frame)
        state = next_state
        eval_reward += reward
        frames_eval.append(next_frame)
        if done:
            gif = False  # Save only the first game of the evaluation as a gif
    print("Evaluation score:\n", eval_reward)
    try:
        generate_gif(frames_eval, total_t, eval_reward, SAVE_MODEL_PATH)
    except IndexError:
        print("No evaluation game finished")
