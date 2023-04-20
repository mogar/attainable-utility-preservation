#!/usr/bin/env python3
import gymnasium as gym

from environments.box import BoxEnvironment
from environments.dog import DogEnvironment

env = BoxEnvironment(render_mode = "human", level = 0)

time_step = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    time_step = env.step(action)

    if time_step.last or time_step.trunc:
        time_step = env.reset()

env.close()