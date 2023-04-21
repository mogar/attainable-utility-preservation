#!/usr/bin/env python3

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class StepResult():
    def __init__(self, observation, reward, last, trunc, info):
        self.observation = observation
        self.last = last
        self.reward = reward
        self.trunc = trunc
        self.info = info

class GridWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="rgb_array", size=0):
        # pygame window size
        self.window_size = 512
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # for human viewing mode, populated the first time that's used
        self.window = None
        self.clock = None

        # basic game setup
        self.size = size
        
        # reward setup
        self.goal_reward = 1

        # basic action setup
        # we have 5 actions {null, right, up, left, down}
        self.action_space = spaces.Discrete(5)

        # dictionary for users
        self.actions = {
            "null": 0,
            "right": 1,
            "up": 2,
            "left": 3,
            "down": 4,
        }

        # map actions to vectors in the state-space
        self._action_to_direction = {
            0: np.array([0, 0]), # null action
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([-1, 0]),
            4: np.array([0, -1]),
        }

    def get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "secret_reward": self.secret_reward,
        }
    
    def _get_hidden_reward(self):
        return self.secret_reward

    def reset(self, seed=None, options=None):
        # in case we ever use the RNG
        super().reset(seed=seed)

        self.terminated = False

        # Set up default environment
        self._walls = None
        self._agent_location = np.array([0, 0])
        self._target_location = np.array([self.size-1, self.size-1])
        
        # Reset rewards
        self.secret_reward = 0
        self.episode_return = 0
        
        # NOTE: render and StepResult return handled in derived class

    def intersects_wall(self, pos):
        return np.any(np.all(self._walls == pos, axis=1))
    
    def render(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Now we draw the walls
        if self._walls is not None:
            for wall in self._walls:
                pygame.draw.rect(
                    canvas,
                    (100, 100, 100),
                    pygame.Rect(
                        pix_square_size * wall,
                        (pix_square_size, pix_square_size),
                    ),
                )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        return canvas

    def _render_frame(self, canvas):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()