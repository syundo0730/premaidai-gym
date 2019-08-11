from math import radians

import gym
import numpy as np
import premaidai_gym

env = gym.make('RoboschoolPremaidAIWalker-v0')

env.reset()

while True:
    action = np.full(env.action_space.shape[0], 0.)
    action[13] = radians(60)  # right arm
    action[18] = radians(-60)  # left arm
    env.step(action)
    env.render()
