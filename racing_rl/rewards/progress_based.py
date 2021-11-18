from typing import Tuple

import gym

from racing_rl import SingleAgentRaceEnv
from racing_rl.envs.track import Track


class ProgressReward(gym.Wrapper):
    def __init__(self, env, track: Track):
        super(ProgressReward, self).__init__(env)
        self._track = track
        self._current_progress = None

    def reset(self, **kwargs):
        obs = super(ProgressReward, self).reset(**kwargs)
        point = obs['pose'][0:2]
        self._current_progress = self._track.get_progress(point)
        return obs

    def step(self, action):
        obs, _, done, info = super(ProgressReward, self).step(action)
        point = obs['pose'][0:2]
        new_progress = info['lap_count'] + self._track.get_progress(point)
        reward = new_progress - self._current_progress
        self._current_progress = new_progress
        return obs, reward, done, info
