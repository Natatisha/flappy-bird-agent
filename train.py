import argparse
import gym
from gym.wrappers import Monitor
import gym_ple
from ddqn import train_ddqn_model
from utils import plot_rewards, save_rewards, load_rewards

DEFAULT_GAMMA = 0.99
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPISODES = 3500


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', action='store', nargs=1, default=[DEFAULT_BATCH_SIZE], type=int,
                        help="Size of a minibatch")
    parser.add_argument('--gamma', '-g', action='store', nargs=1, default=[DEFAULT_GAMMA], type=float,
                        help="Discount factor")
    parser.add_argument('--num_episodes', '-n', action='store', nargs=1, default=[DEFAULT_NUM_EPISODES], type=int,
                        help="Number of episodes")
    parser.add_argument('--record', '-r', action='store', nargs=1, default=[False], type=bool,
                        help="Whether or not to record a video of training. Default=False")
    parser.add_argument('--record_frequency', '-f', action='store', nargs=1, default=[100], type=int,
                        help="How ofter to record video. "
                             "E.g. if set to 10, then video would be recorded each 10-th episode."
                             "If '--record' argument is set to False this parameter would be ignored. "
                             "Default=100")
    parser.add_argument('--model_out', action='store', nargs=1, default=['ddqn_weights.npz'],
                        help="Name of the file with saved wights of the trained model. Should have .npz extension. "
                             "Default is 'ddqn_weights.npz'")
    parser.add_argument('--rewards_out', action='store', nargs=1, default=['ddqn_rewards.npy'],
                        help="Name of the file with saved rewards obtained during training. Should have .npy extension."
                             " Default is 'ddqn_rewards.npy'")
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = get_arguments()

    batch_size = args['batch_size'][0]
    gamma = args['gamma'][0]
    num_episodes = args['num_episodes'][0]
    record_video = args['record'][0]
    record_each = args['record_frequency'][0]
    model_out = args['model_out'][0]
    rewards_out = args['rewards_out'][0]

    env = gym.make("PongDeterministic-v4")

    model, episode_rewards = train_ddqn_model(env, num_episodes, batch_size, gamma)
    # save_rewards(episode_rewards, file_name=rewards_out)
