from typing import Tuple

import gym

from racing_rl import SingleAgentRaceEnv
from racing_rl.envs.track import Track


class MinSteeringReward(gym.Wrapper):
    def __init__(self, env, track: Track):
        super(MinSteeringReward, self).__init__(env)

    def step(self, action):
        obs, _, done, info = super(MinSteeringReward, self).step(action)
        if done:
            reward = - 10
        else:
            reward = 1 - action['steering'] ** 2
        return obs, reward, done, info
