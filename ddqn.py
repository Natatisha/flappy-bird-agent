import sys

import tensorflow as tf
import numpy as np
from datetime import datetime

from image_transformer import ImageTransformer
from replay_buffer import ReplayBuffer

# constants
# for testing
# MAX_EXPERIENCES = 10000
# MIN_EXPERIENCES = 100

# prod
MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 10000
TARGET_UPD_PERIOD = 10000
IMG_SIZE = 80
ACTIONS_NUM = 2
FRAMES_IN_STATE = 4


class DDQN:

    def __init__(self, actions_n, conv_layers_sizes, hidden_layers_sizes, scope, frame_shape=(IMG_SIZE, IMG_SIZE),
                 state_depth=FRAMES_IN_STATE):
        self.actions_n = actions_n
        self.scope = scope

        with tf.variable_scope(self.scope):
            self.X = tf.placeholder(dtype=tf.float32, shape=(None, frame_shape[1], frame_shape[0], state_depth),
                                    name='X')
            self.actions = tf.placeholder(dtype=tf.int32, shape=(None,), name='actions')
            self.G = tf.placeholder(dtype=tf.float32, shape=(None,), name='G')

            Z = self.X / 255.

            for num_output_filters, filtersz, poolsz in conv_layers_sizes:
                Z = tf.contrib.layers.conv2d(
                    Z,
                    num_output_filters,
                    filtersz,
                    poolsz,
                    activation_fn=tf.nn.relu
                )

            # fully connected layers
            Z = tf.contrib.layers.flatten(Z)
            for M in hidden_layers_sizes:
                Z = tf.contrib.layers.fully_connected(Z, M)

            # final output layer
            self.predict_op = tf.contrib.layers.fully_connected(Z, actions_n)

            selected_action_values = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions, actions_n), axis=1)

            cost = tf.reduce_mean(tf.losses.huber_loss(self.G, selected_action_values))
            self.train_optimizer = tf.train.AdamOptimizer(1e-5).minimize(cost)
            self.cost = cost

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
        return self.sess.run(self.predict_op, feed_dict={self.X: X})

    def train(self, states, actions, targets):
        cost, _ = self.sess.run([self.cost, self.train_optimizer], feed_dict={
            self.actions: actions,
            self.G: targets,
            self.X: states
        })
        return cost

    def sample_action(self, states, eps):
        if np.random.random() < eps:
            return np.random.choice(self.actions_n)
        else:
            return np.argmax(self.predict([states])[0])

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

    next_Qs = target_model.predict(next_states)
    next_Q = np.amax(next_Qs, axis=1)
    # if terminal state - target is reward, else r + gamma*max(next_Q)
    targets = rewards + gamma * next_Q * np.invert(dones).astype(np.float32)

    loss = model.train(states, actions, targets)
    return loss


def decaying_epsilon(x, min_value, decay_rate=0.99):
    eps = decay_rate ** x
    return max(eps, min_value)


def play_one_episode(
        env,
        session,
        total_t,
        episode_num,
        model,
        target_model,
        replay_buffer,
        image_tansformer,
        gamma,
        epsilon,
        epsilon_change,
        epsilon_min=0.1,
        target_upd_period=TARGET_UPD_PERIOD):
    t0 = datetime.now()
    total_training_time = 0

    raw_frame = env.reset()
    frame = image_tansformer.transform(raw_frame, session)
    state = np.stack([frame] * 4, axis=2)
    loss = None

    done = False

    episode_reward = 0
    num_steps = 0

    while not done:

        if total_t % target_upd_period == 0:
            target_model.copy_from(model)
            print("Copied model parameters to target network. total_t = %s, period = %s" % (
                total_t, target_upd_period))

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

        epsilon = max(epsilon - epsilon_change, epsilon_min)

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

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_change = (epsilon - epsilon_min) / 300000

    # Create models
    conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    hidden_layer_sizes = [512]

    model = DDQN(
        ACTIONS_NUM,
        conv_layer_sizes,
        hidden_layer_sizes,
        scope="model")

    target_model = DDQN(
        ACTIONS_NUM,
        conv_layer_sizes,
        hidden_layer_sizes,
        scope="target_model"
    )
    image_transformer = ImageTransformer(out_shape=(IMG_SIZE, IMG_SIZE))

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
                i,
                model,
                target_model,
                replay_buffer,
                image_transformer,
                gamma,
                epsilon,
                epsilon_change,
                epsilon_min,
            )
            episode_rewards[i] = episode_reward

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
