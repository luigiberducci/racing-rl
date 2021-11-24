from typing import Tuple

import gym

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

    def _compute_progress(self, obs, info):
        assert 'pose' in obs and 'lap_count' in info
        point = obs['pose'][0:2]
        progress = info['lap_count'] + self._track.get_progress(point)
        return progress

    def step(self, action):
        obs, _, done, info = super(ProgressReward, self).step(action)
        new_progress = self._compute_progress(obs, info)
        if done:
            reward = - 10
        else:
            reward = 100 * new_progress - self._current_progress
        self._current_progress = new_progress
        return obs, reward, done, info
