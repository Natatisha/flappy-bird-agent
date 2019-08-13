import argparse
import tensorflow as tf
import numpy as np
from image_transformer import ImageTransformer
from ddqn import update_state
from utils import generate_gif, plot_rewards

from environment import FlappyBirdWrapper
from q_learning import QLearningAgent

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport

DEFAULT_GIF_PATH = 'gifs/'
DEFAULT_NUM_EVAL_EPISODES = 100
DEFAULT_MODEL_TYPE = "dqn"
MAX_EPISODE_SCORE = 300

Path(DEFAULT_GIF_PATH).mkdir(exist_ok=True)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', action='store', nargs=1, required=False, default=DEFAULT_MODEL_TYPE,
                        help="Model type. Can be q_learn or dqn")
    parser.add_argument('--model_path', '-path', action='store', nargs=1,
                        help="Path to the folder where to store the output gif")
    parser.add_argument('--gif_path', action='store', nargs=1, default=[DEFAULT_GIF_PATH],
                        help="Path to the folder where to store the output gif")
    parser.add_argument('--num_episodes', '-n', action='store', nargs=1, default=[DEFAULT_NUM_EVAL_EPISODES], type=int,
                        help="Number of episodes")
    return vars(parser.parse_args())


def evaluate(env, action_fun, num_episodes, gif_path):
    total_t = 0
    gif = True
    eval_rewards = []
    best_reward = -100
    for episode_i in range(num_episodes):
        frames_for_gif = []
        done = False
        state = env.reset()
        episode_reward_sum = 0

        while not done and episode_reward_sum <= MAX_EPISODE_SCORE:
            action = action_fun(state)
            state, reward, done, _ = env.step(action, False)
            episode_reward_sum += reward
            frames_for_gif.append(env.get_screen(with_score=True))
            total_t += 1

            if done:
                # generate git only if this score beats the best
                gif = episode_reward_sum > best_reward
                best_reward = max(best_reward, episode_reward_sum)
                eval_rewards.append(episode_reward_sum)
                print("Evaluation score:\n", episode_reward_sum)

        if gif:
            try:
                generate_gif(frames_for_gif, total_t, episode_reward_sum, gif_path)
            except IndexError:
                print("No evaluation game finished")

    return eval_rewards


if __name__ == "__main__":
    args = get_arguments()

    model_type = args['model'][0]
    num_episodes = args['num_episodes'][0]
    gif_path = args['gif_path'][0]
    model_path = args['model_path'][0] if 'model_path' in args and args['model_path'] else None

    eval_rewards = []

    if model_type == "q_learn":
        env = FlappyBirdWrapper(screen_output=False)
        model = QLearningAgent(env)
        model.q_table = model.load(model_path) if model_path is not None else model.load()


        def action_fun(state):
            return model.sample_action(state, 0.)


        eval_rewards = evaluate(env, action_fun, num_episodes, gif_path)
    elif model_type == "dqn":
        env = FlappyBirdWrapper(screen_output=False)
        model_path = model_path if model_path is not None else "dqn_outputs/my_model-601118.meta"
        model_folder = model_path.split("/")[0] + "/"
        saver = tf.train.import_meta_graph(model_path)
        scope = "model"

        with tf.get_default_graph().as_default():
            collection = tf.GraphKeys.TRAINABLE_VARIABLES
            variables = tf.get_collection(collection, scope=scope)
            assert len(variables) > 0
            print("Variables in scope '{}':".format(scope))
            for v in variables:
                print("\t" + str(v))

            X = tf.get_default_graph().get_tensor_by_name(scope + '/X:0')
            w1 = tf.get_default_graph().get_tensor_by_name(scope + '/Variable:0')
            b1 = tf.get_default_graph().get_tensor_by_name(scope + '/Variable_1:0')
            w2 = tf.get_default_graph().get_tensor_by_name(scope + '/Variable_2:0')
            b2 = tf.get_default_graph().get_tensor_by_name(scope + '/Variable_3:0')
            w3 = tf.get_default_graph().get_tensor_by_name(scope + '/Variable_4:0')
            b3 = tf.get_default_graph().get_tensor_by_name(scope + '/Variable_5:0')

            a1 = tf.matmul(X, w1) + b1
            z1 = tf.nn.tanh(a1)

            a2 = tf.matmul(z1, w2) + b2
            z2 = tf.nn.tanh(a2)

            q_values = tf.matmul(z2, w3) + b3

        with tf.Session() as session:
            saver.restore(session, tf.train.latest_checkpoint(model_folder))


            def action_fun(state):
                prediction = session.run(q_values, feed_dict={X: np.atleast_2d(state)})
                action = np.argmax(prediction[0])
                return action


            eval_rewards = evaluate(env, action_fun, num_episodes, gif_path)
    else:
        raise RuntimeWarning("Model {} is not supported!".format(model_type))

    eval_score = np.mean(eval_rewards)
    print("MEAN SCORE {}".format(eval_score))
    plot_rewards(eval_rewards)
