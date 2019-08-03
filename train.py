import argparse
from ddqn import train_ddqn_model
from q_learning import train_q_learning_model
from dqn_simple import train_dqn

DEFAULT_GAMMA = 0.99
DEFAULT_BATCH_SIZE = 32
DEFAULT_MODEL = 'ddqn'


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', action='store', nargs=1, default=[DEFAULT_BATCH_SIZE], type=int,
                        help="Size of a minibatch")
    parser.add_argument('--gamma', '-g', action='store', nargs=1, default=[DEFAULT_GAMMA], type=float,
                        help="Discount factor")
    parser.add_argument('--model', '-m', action='store', nargs=1, default=[DEFAULT_MODEL],
                        help="Model type. Available types: ddqn, q_learn, dqn")
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = get_arguments()

    batch_size = args['batch_size'][0]
    gamma = args['gamma'][0]
    model = args['model'][0]
    # record_video = args['record'][0]
    # record_each = args['record_frequency'][0]
    # model_out = args['model_out'][0]
    # rewards_out = args['rewards_out'][0]

    if model == 'ddqn':
        model, episode_rewards = train_ddqn_model(batch_size, gamma)
    elif model == 'q_learn':
        train_q_learning_model(gamma)
    elif model == 'dqn':
        train_dqn(gamma, batch_size)
    else:
        raise RuntimeWarning("Model {} is not supported!".format(model))
