import os
import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
import math
import cv2


class FlappyBirdWrapper(object):

    def __init__(self, screen_output=True, image_transformer=None, agent_history_length=4, processed_reward=True):
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
        self.process_reward = processed_reward
        self.low_obs_values = (-15, -16, 0, 75)
        self.high_obs_values = (300, 10, 310, 245)
        self.score = 0.
        self.sess = None

    def set_session(self, sess):
        self.sess = sess

    def reset(self):
        self.score = 0
        self.env.reset_game()
        out_state, _ = self.process_state()
        state = np.stack([out_state] * 4, axis=2) if self.screen_out else out_state
        return state

    def process_state(self):
        if self.screen_out:
            raw_state = self.get_screen()
            if self.image_transformer is not None:
                assert self.sess
                state = self.image_transformer.transform(raw_state, sess=self.sess)
            else:
                state = raw_state
        else:
            state, raw_state = self.get_meta_state()
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
        next_pipe_center_y = next_pipe_top_y + (next_pipe_bottom_y - next_pipe_top_y) / 2

        return (player_y, player_vel, next_pipe_dist_to_player, next_pipe_center_y)

    def _process_reward(self, reward):
        if self.process_reward:
            state, _ = self.get_meta_state()
            dist = self._calc_dist(state[0], state[3], state[2] + 50.)  # shift target point by 50 pixels to the right
            if reward == 0:
                reward = 0.95 ** dist
            elif reward < 0:
                reward = -300
            else:
                reward = 5

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
            reward = self._process_reward(reward)
        self.score += reward
        return processed_new_state, reward, done, new_state

    def close(self):
        pass  # if we'll decide to change the environment, for example to OpenAI gym, we'll need this function

    def get_screen(self, with_score=False):
        img = self._rotate_and_flip_img(self.env.getScreenRGB())
        if with_score:
            text = np.zeros(shape=img.shape, dtype=type(img[0][0][0]))
            cv2.putText(text, "Score: {}".format(self.score), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            img = cv2.addWeighted(img, 1, text, 1., 0)
        return img

    def get_meta_state(self):
        raw_state = self.env.getGameState()
        state = self._process_dict_state(raw_state)
        return state, raw_state
