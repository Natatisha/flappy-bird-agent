import os
import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
import math


class FlappyBirdWrapper(object):

    def __init__(self, screen_output=True, image_transformer=None, agent_history_length=4):
        os.environ["SDL_VIDEODRIVER"] = "dummy"  # to avoid empty game window creation
        self.game = FlappyBird()
        self.env = PLE(self.game, display_screen=False, reward_values={"loss": -5.0, "positive": 1.0})
        self.env.init()
        self.screen_out = screen_output
        self.image_transformer = image_transformer
        self.state = None
        self.agent_history_length = agent_history_length
        self.actions = self.game.actions
        self.observations_num = 4
        self.sess = None

    def set_session(self, sess):
        self.sess = sess

    def reset(self):
        self.env.reset_game()
        out_state, _ = self.process_state()
        state = np.stack([out_state] * 4, axis=2) if self.screen_out else out_state
        return state

    def process_state(self):
        if self.screen_out:
            raw_state = self._rotate_and_flip_img(self.env.getScreenRGB())
            if self.image_transformer is not None:
                assert self.sess
                state = self.image_transformer.transform(raw_state, sess=self.sess)
            else:
                state = raw_state
        else:
            raw_state = self.env.getGameState()
            state = self._process_dict_state(raw_state)
        return state, raw_state

    def _process_dict_state(self, raw_state):
        player_y = raw_state["player_y"]
        player_vel = raw_state["player_vel"]
        next_pipe_bottom_y = raw_state["next_pipe_bottom_y"]
        next_pipe_top_y = raw_state["next_pipe_top_y"]
        next_next_pipe_bottom_y = raw_state["next_next_pipe_bottom_y"]
        next_next_pipe_top_y = raw_state["next_next_pipe_top_y"]
        next_pipe_dist_to_player = raw_state["next_pipe_dist_to_player"]
        next_next_pipe_dist_to_player = raw_state["next_next_pipe_dist_to_player"]
        # dist = self._calc_dist(player_y, next_pipe_bottom_y, next_pipe_top_y, next_pipe_dist_to_player)
        # dist_next = self._calc_dist(player_y, next_next_pipe_bottom_y, next_next_pipe_top_y,
        #                             next_next_pipe_dist_to_player)
        next_pipe_center_y = next_pipe_top_y + (next_pipe_bottom_y - next_pipe_top_y) / 2
        next_next_pipe_center_y = next_next_pipe_top_y + (next_next_pipe_bottom_y - next_next_pipe_top_y) / 2

        return (player_y, player_vel, next_pipe_dist_to_player, next_pipe_center_y)

    def _process_reward(self, state, reward):
        if not self.screen_out:
            dist = self._calc_dist(state[0], state[3], state[2])
            if reward == 0:
                reward = 0.99 ** dist
            elif reward < 0:
                reward = -300
        return reward

    def _process_action(self, action):
        return self.actions['up'] if action > 0 else action

    @staticmethod
    def _calc_dist(player_y, next_pipe_center_y, dist_x):
        dist_y = next_pipe_center_y - player_y
        dist = math.sqrt(dist_x ** 2 + dist_y ** 2)
        return dist

    @staticmethod
    def _rotate_and_flip_img(img):
        return np.fliplr(np.rot90(img, 3))

    def step(self, action, train=True):
        action = self._process_action(action)
        reward = self.env.act(action)
        processed_new_state, new_state = self.process_state()
        done = self.env.game_over()
        if train:
            reward = self._process_reward(processed_new_state, reward)
        return processed_new_state, reward, done, new_state

    def close(self):
        pass  # if we'll decide to change environment, for example to OpenAI gym, we'll need this function
