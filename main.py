import gym
from gym.wrappers import Monitor
import gym_ple
from ddqn import train_ddqn_model
from visualization import plot_rewards

if __name__ == '__main__':
    gamma = 0.99
    batch_size = 32
    num_episodes = 100

    env = gym.make("FlappyBird-v0")
    env = Monitor(env, "./ddpq_videos", video_callable=lambda episode_id: episode_id % 10 == 0, force=True)

    model, episode_rewards = train_ddqn_model(env, num_episodes, batch_size, gamma, True, 1)
    plot_rewards(episode_rewards)
