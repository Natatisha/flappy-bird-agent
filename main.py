import gym
from gym.wrappers import Monitor
import gym_ple
from ddqn import train_ddqn_model
from utils import plot_rewards, save_rewards, load_rewards

if __name__ == '__main__':
    gamma = 0.99
    batch_size = 128
    num_episodes = 5000

    env = gym.make("FlappyBird-v0")
    env = Monitor(env, "./ddpq_videos", video_callable=lambda episode_id: episode_id % 10 == 0, force=True)

    model, episode_rewards = train_ddqn_model(env, num_episodes, batch_size, gamma)
    save_rewards(episode_rewards)
