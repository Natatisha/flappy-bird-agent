import os
import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
import cv2
import math
import gym
import gym_ple


class FlappyBirdWrapper(object):

    def __init__(self, screen_output=True, image_transformer=None, agent_history_length=4):
        os.environ["SDL_VIDEODRIVER"] = "dummy"  # to avoid empty game window creation
        # self.game = FlappyBird()
        # self.env = PLE(self.game, display_screen=False, reward_values={"loss": -5.0, "positive": 5.0})
        self.env = gym.make('FlappyBird-v0')
        # self.env.init()
        self.screen_out = screen_output
        self.image_transformer = image_transformer
        # self.state = None
        self.agent_history_length = agent_history_length
        # self.actions = self.game.actions
        self.observations_num = 4
        self.sess = None

    def set_session(self, sess):
        self.sess = sess

    def reset(self):
        # self.env.reset_game()
        # out_state, _ = self.process_state()
        out_state = self.image_transformer.transform(self.env.reset(),
                                                     sess=self.sess)
        state = np.stack([out_state] * 4, axis=2) if self.screen_out else out_state
        return state

    def process_state(self):
        if self.screen_out and self.sess is not None:
            raw_state = self._rotate_and_flip_img(self.env.getScreenRGB())
            state = self.image_transformer.transform(raw_state,
                                                     sess=self.sess) if self.image_transformer is not None else raw_state
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
        dist = self._calc_dist(player_y, next_pipe_bottom_y, next_pipe_top_y, next_pipe_dist_to_player)
        dist_next = self._calc_dist(player_y, next_next_pipe_bottom_y, next_next_pipe_top_y,
                                    next_next_pipe_dist_to_player)
        return (player_y, player_vel, dist, dist_next)

    def _process_reward(self, state, reward):
        if reward == 0:
            return np.tanh((1. / state[2])) * 100.
        else:
            return reward

    @staticmethod
    def _calc_dist(player_y, next_pipe_bottom_y, next_pipe_top_y, dist_x):
        pipe_window_center = (next_pipe_bottom_y - next_pipe_top_y) / 2.
        dist_y = pipe_window_center - player_y
        dist = math.sqrt(dist_x ** 2 + dist_y ** 2)
        return dist

    @staticmethod
    def _rotate_and_flip_img(img):
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        new_center = (h / 2, w / 2)

        rotation_mat = cv2.getRotationMatrix2D(center, 270, 1.)

        rotation_mat[0, 2] += new_center[0] - center[0]
        rotation_mat[1, 2] += new_center[1] - center[1]

        result = cv2.warpAffine(img, rotation_mat, (h, w))
        result = cv2.flip(result, 1)
        return np.fliplr(np.rot90(img, 3))  # TODO test this

    def step(self, action):
        new_frame, reward, done, _ = self.env.step(action)
        processed_new_frame = self.image_transformer.transform(new_frame, self.sess)
        # processed_new_frame, new_frame = self.process_state()

        # done = self.env.game_over()
        # if self.screen_out:
        #     new_state = np.append(self.state[:, :, 1:], np.expand_dims(processed_new_frame, 2), axis=2)
        # else:
        #     new_state = processed_new_frame
        #     reward = self._process_reward(new_state, reward)
        # self.state = new_state
        # print("{} {} {}".format(action, new_state, reward))
        return processed_new_frame, reward, done, new_frame

    def close(self):
        pass  # if we'll decide to change environment, for example to OpenAI gym, we'll need this function