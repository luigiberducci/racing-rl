import gym
import numpy as np


class MinActionReward(gym.Wrapper):
    def __init__(self, env, collision_penalty: float = 0.0):
        assert collision_penalty >= 0.0, f"penalty must be >=0 and will be subtracted to the reward ({collision_penalty}"
        self._collision_penalty = collision_penalty
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
            reward = - self._collision_penalty
        else:
            reward = 1 - (1 / len(action) * np.linalg.norm(action) ** 2)
        return obs, reward, done, info
