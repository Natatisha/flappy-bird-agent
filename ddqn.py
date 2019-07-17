import sys

import tensorflow as tf
import numpy as np
from datetime import datetime
from enum import Enum
import math

from image_transformer import ImageTransformer
from replay_buffer import ReplayBuffer


class EpsilonDecay(Enum):
    LINEAR = 0
    SINUSOID = 1
    EXPONENTIAL = 2


# constants
# for testing
MAX_EXPERIENCES = 10000
MIN_EXPERIENCES = 100

# prod
# Flappy
# EPSILON_DECAY_TYPE = EpsilonDecay.LINEAR
# EPSILON_INITIAL = 1.
# EPSILON_CHECKPOINT = 0.1
# EPSILON_FINAL = 0.001
# EPSILON_ANNEALING_FRAMES = 1e6
# MAX_FRAMES = 2e6

# Pong
EPSILON_DECAY_TYPE = EpsilonDecay.LINEAR
EPSILON_INITIAL = 1.
EPSILON_CHECKPOINT = 0.1
EPSILON_FINAL = 0.01
EPSILON_ANNEALING_FRAMES = 1e6
MAX_FRAMES = 2e6

# Breakout
# EPSILON_DECAY_TYPE = EpsilonDecay.LINEAR
# EPSILON_INITIAL = 1.
# EPSILON_CHECKPOINT = 0.1
# EPSILON_FINAL = 0.001
# EPSILON_ANNEALING_FRAMES = 1e6
# MAX_FRAMES = 2e6

# MAX_EXPERIENCES = 500000
# MIN_EXPERIENCES = 50000

TARGET_UPD_PERIOD = 10000
IMG_SIZE = 80
ACTIONS_NUM = 4
FRAMES_IN_STATE = 4
SAVE_EACH = 1000


class DDQN:

    def __init__(self, actions_n, hidden_layers_size, scope, learning_rate=1e-6, frame_shape=(IMG_SIZE, IMG_SIZE),
                 agent_history_length=FRAMES_IN_STATE):
        self.actions_n = actions_n
        self.scope = scope
        self.lr = learning_rate

        with tf.variable_scope(self.scope):
            self.X = tf.placeholder(dtype=tf.float32,
                                    shape=(None, frame_shape[0], frame_shape[1], agent_history_length),
                                    name='X')
            self.X_scaled = self.X / 255

            self.conv1 = tf.layers.conv2d(self.X_scaled, filters=32, kernel_size=[8, 8], strides=4,
                                          kernel_initializer=tf.variance_scaling_initializer(scale=2), padding='valid',
                                          activation=tf.nn.relu, use_bias=False, name='conv1')
            self.conv2 = tf.layers.conv2d(self.conv1, filters=64, kernel_size=[4, 4], strides=2,
                                          kernel_initializer=tf.variance_scaling_initializer(scale=2), padding='valid',
                                          activation=tf.nn.relu, use_bias=False, name='conv2')
            self.conv3 = tf.layers.conv2d(self.conv2, filters=64, kernel_size=[3, 3], strides=1,
                                          kernel_initializer=tf.variance_scaling_initializer(scale=2), padding='valid',
                                          activation=tf.nn.relu, use_bias=False, name='conv3')
            self.conv4 = tf.layers.conv2d(self.conv3, filters=hidden_layers_size, kernel_size=[7, 7], strides=1,
                                          kernel_initializer=tf.variance_scaling_initializer(scale=2), padding='valid',
                                          activation=tf.nn.relu, use_bias=False, name='conv4')
            print(tf.shape(self.conv4))

            self.advantagestream, self.valuestream = tf.split(self.conv4, 2, 3)

            self.advantagestream = tf.layers.flatten(self.advantagestream)
            self.valuestream = tf.layers.flatten(self.valuestream)

            self.advantage = tf.layers.dense(self.advantagestream, self.actions_n,
                                             kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                             name='advantage')
            self.value = tf.layers.dense(self.valuestream, 1,
                                         kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                         name='value')

            self.q_values = self.value + tf.subtract(self.advantage,
                                                     tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
            self.best_action = tf.argmax(self.q_values, axis=1)

            self.actions = tf.placeholder(dtype=tf.int32, shape=(None,), name='actions')
            self.target_q = tf.placeholder(dtype=tf.float32, shape=(None,), name='target_q')

            self.Q = tf.reduce_sum(self.q_values * tf.one_hot(self.actions, actions_n), axis=1)
            self.cost = tf.reduce_mean(tf.losses.huber_loss(labels=self.target_q, predictions=self.Q))
            self.train_optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

            self.sess = None

    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)

        other = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        other = sorted(other, key=lambda v: v.name)

        ops = []
        for p, q in zip(mine, other):
            op = p.assign(q)
            ops.append(op)

        self.sess.run(ops)

    def set_session(self, session):
        self.sess = session

    def predict(self, X):
        return self.sess.run(self.q_values, feed_dict={self.X: X})

    def best_action(self, X):
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
            return self.best_action([state])[0]

    def save(self, file_name='tf_dqn_weights.npz'):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        params = self.sess.run(params)
        np.savez(file_name, *params)

    def load(self, file_name='tf_dqn_weights.npz'):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        npz = np.load(file_name)
        ops = []
        for p, (_, v) in zip(params, npz.iteritems()):
            ops.append(p.assign(v))
        self.sess.run(ops)


def update_state(state, new_frame):
    return np.append(state[:, :, 1:], np.expand_dims(new_frame, 2), axis=2)


def learn(model, target_model, replay_buffer, gamma):
    states, actions, rewards, next_states, dones = replay_buffer.sample()

    best_actions = model.best_action(next_states)
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
        loss = learn(model, target_model, replay_buffer, gamma)
        dt = datetime.now() - t0_2

        total_training_time += dt.total_seconds()
        num_steps += 1
        total_t += 1

        state = next_state

    return total_t, episode_reward, (datetime.now() - t0).total_seconds(), \
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


def train_ddqn_model(env, num_episodes, batch_size, gamma, weights_file_name='ddqn_weights.npz'):
    replay_buffer = ReplayBuffer(MAX_EXPERIENCES, batch_size, (IMG_SIZE, IMG_SIZE), FRAMES_IN_STATE)
    episode_rewards = np.zeros(num_episodes)

    # Create models
    hidden_layer_sizes = [1024]

    model = DDQN(
        ACTIONS_NUM,
        hidden_layer_sizes,
        scope="model")

    target_model = DDQN(
        ACTIONS_NUM,
        hidden_layer_sizes,
        scope="target_model")

    image_transformer = ImageTransformer(out_shape=(IMG_SIZE, IMG_SIZE), crop_boundaries=(34, 0, 160, 160))
    epsilon_scheduler = EpsilonGreedyScheduler(EpsilonDecay.LINEAR)

    total_t = 0

    with tf.Session() as sess:
        model.set_session(sess)
        target_model.set_session(sess)
        sess.run(tf.global_variables_initializer())

        populate_experience(env, image_transformer, replay_buffer, sess)

        # Play a number of episodes and learn!
        t0 = datetime.now()
        for i in range(num_episodes):
            total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one_episode(
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

            if (i + 1) % SAVE_EACH == 0:
                model.save(file_name=weights_file_name)

            last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
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

        model.save(file_name=weights_file_name)

    return model, episode_rewards
