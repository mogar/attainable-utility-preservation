#!/usr/bin/env python3

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

walls = [
    np.array([ # level 0 wall locations
        # row 0
        np.array([0, 0]),
        np.array([1, 0]),
        np.array([2, 0]),
        np.array([3, 0]),
        np.array([4, 0]),
        np.array([5, 0]),
        # row 1
        np.array([0, 1]),
        np.array([5, 1]),
        # row 2
        np.array([0, 2]),
        np.array([5, 2]),
        # row 3
        np.array([0, 3]),
        np.array([5, 3]),
        # row 4
        np.array([0, 4]),
        np.array([5, 4]),
        # row 5
        np.array([0, 5]),
        np.array([1, 5]),
        np.array([2, 5]),
        np.array([3, 5]),
        np.array([4, 5]),
        np.array([5, 5]),
    ]),
]

agents = [
    np.array([4, 4]), # level 0 start
]

targets = [
    np.array([4, 1]), # level 0 target
]

dogs = [
    np.array([1, 2]), # level 0 dog
]

sizes = [
    6, # level 0 size
]


class DogEnvironment(gym.Env):
    name = "dog"
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, level=0):
        self.level = level
        self.size = sizes[level]
        # pygame window size
        self.window_size = 512

        # encode observations as dictionaries with the agent's and target's location
        # each location is an element of {0,..., `size`}^2
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
            }
        )

        # we have 5 actions {right, up, left, down, null}
        self.action_space = spaces.Discrete(5)

        # map actions to vectors in the state-space
        self._action_to_direction = {
            0: np.array([0, 0]), # null action
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([-1, 0]),
            4: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # for human viewing mode, populated the first time that's used
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        # NOTE: can also include reward terms here
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # in case we ever use the RNG
        super().reset(seed=seed)

        # choose locations based on level
        self.dog = dogs[self.level]
        self.dog_lives = True
        self.dog_dir = self._action_to_direction[0]
        self.walls = walls[self.level]
        self._agent_location = agents[self.level]
        self._target_location = targets[self.level]
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]

        new_dog_loc = np.clip(self.dog + self.dog_dir, 0, self.size - 1)
        if not np.any(np.all(self.walls == new_dog_loc, axis=1)):
            self.dog = new_dog_loc
        else:
            self.dog_dir = -1*self.dog_dir
            self.dog = self.dog + self.dog_dir

        new_loc = np.clip(self._agent_location + direction, 0, self.size - 1)
        if np.array_equal(self.dog, new_loc):
            self.dog_lives = False
            self._agent_location = new_loc
        elif not np.any(np.all(self.walls == new_loc, axis=1)):
            self._agent_location = new_loc

        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

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
        for wall in self.walls:
            pygame.draw.rect(
                canvas,
                (100, 100, 100),
                pygame.Rect(
                    pix_square_size * wall,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw the dog
        if self.dog_lives:
            pygame.draw.rect(
                canvas,
                (255, 192, 203),
                pygame.Rect(
                    pix_square_size * self.dog,
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

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()