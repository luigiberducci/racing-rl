from typing import Tuple

import gym
import numpy as np

from racing_rl import SingleAgentRaceEnv
from racing_rl.envs.track import Track


class MinSteeringReward(gym.Wrapper):
    # todo: it should be replaced by the more-general min-action
    def __init__(self, env):
        super(MinSteeringReward, self).__init__(env)

    def step(self, action):
        obs, _, done, info = super(MinSteeringReward, self).step(action)
        steering_low = self.action_space['steering'].low
        steering_high = self.action_space['steering'].high
        norm_steering = 2 * (action['steering'] - steering_low) / (steering_high - steering_low) - 1
        if done:
            reward = - 10
        else:
            reward = 1 - (action['steering'] ** 2)
        return obs, reward, done, info


class MinActionReward(gym.Wrapper):
    def __init__(self, env):
        super(MinActionReward, self).__init__(env)

    def _normalize_action(self, name, action_val):
        low, high = self.action_space[name].low, self.action_space[name].high
        norm_action = 2 * (action_val - low) / (high - low) - 1
        return norm_action

    def step(self, action):
        obs, _, done, info = super(MinActionReward, self).step(action)
        action = [self._normalize_action(key, val) for key, val in action.items()]
        assert all(abs(a) <= 1 for a in action)
        if done:
            reward = - 10
        else:
            reward = 1 - (1/len(action) * np.linalg.norm(action) ** 2)
        return obs, reward, done, info
