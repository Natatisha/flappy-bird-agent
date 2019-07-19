import os
import argparse
import gym
import gym_ple
import tensorflow as tf
import numpy as np
from image_transformer import ImageTransformer
from ddqn import update_state
from utils import generate_gif

DEFAULT_GIF_PATH = 'gifs/'
DEFAULT_NUM_EVAL_EPISODES = 100

os.makedirs(DEFAULT_GIF_PATH, exist_ok=True)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', action='store', nargs=1, required=True,
                        help="Name of a saved model")
    parser.add_argument('--gif', action='store', nargs=1, default=[True], type=bool,
                        help="Whether too write gif of the first evaluation episode. Default=True")
    parser.add_argument('--gif_path', action='store', nargs=1, default=[DEFAULT_GIF_PATH],
                        help="Path to the folder where to store the output gif")
    parser.add_argument('--num_episodes', '-n', action='store', nargs=1, default=[DEFAULT_NUM_EVAL_EPISODES], type=int,
                        help="Number of episodes")
    return vars(parser.parse_args())


OBS_SHAPE = (210, 160, 3)
CROP_BOUNDS = (30, 0, 170, 160)
IMG_SIZE = 84
FRAMES_IN_STATE = 4
SAVE_MODEL_PATH = "outputs/"


def evaluate_model(env, file_name, num_episodes, gif, gif_path):
    saver = tf.train.import_meta_graph(SAVE_MODEL_PATH + file_name)
    scope = "model"

    with tf.get_default_graph().as_default():
        X = tf.get_default_graph().get_tensor_by_name(scope + '/X:0')
        q_values = tf.get_default_graph().get_tensor_by_name(scope + '/q_values:0')
        best_action = tf.argmax(q_values, axis=1)

    frames_for_gif = []
    eval_rewards = np.empty(num_episodes)
    evaluate_frame_number = 0
    done = False

    image_transformer = ImageTransformer(origin_shape=OBS_SHAPE, out_shape=(IMG_SIZE, IMG_SIZE),
                                         crop_boundaries=CROP_BOUNDS)

    with tf.Session() as session:
        saver.restore(session, tf.train.latest_checkpoint(SAVE_MODEL_PATH))
        for i in range(num_episodes):
            episode_reward = 0
            obs = env.reset()
            frames_for_gif.append(obs)
            frame = image_transformer.transform(obs, session)
            state = np.stack([frame] * FRAMES_IN_STATE, axis=2)
            done = False

            while not done:
                action = session.run(best_action, feed_dict={X: [state]})[0]
                next_frame, reward, done, _ = env.step(action)
                frame = image_transformer.transform(next_frame, session)
                next_state = update_state(state, frame)
                state = next_state

                episode_reward += reward
                evaluate_frame_number += 1

                if gif:
                    frames_for_gif.append(next_frame)
                if done:
                    eval_rewards[i] = episode_reward
                    gif = False  # Save only the first game of the evaluation as a gif

    print("Evaluation score:\n", np.mean(eval_rewards))
    try:
        generate_gif(frames_for_gif, eval_rewards[0], gif_path)
    except IndexError:
        print("No evaluation game finished")


if __name__ == "__main__":
    args = get_arguments()

    num_episodes = args['num_episodes'][0]
    model_name = args['name'][0]
    gif = args['gif'][0]
    gif_path = args['gif_path'][0]

    env = gym.make("PongDeterministic-v4")

    evaluate_model(env, model_name, num_episodes, gif, gif_path)
