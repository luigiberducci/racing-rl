from typing import Tuple

import gym

from racing_rl.envs.track import Track
from racing_rl.rewards.core.reward import RewardFunction


class ProgressReward(RewardFunction):

    def __init__(self, track, collision_penalty: float = 0.0):
        assert collision_penalty >= 0.0, f"penalty must be >=0 and will be subtracted to the reward ({collision_penalty}"
        self._collision_penalty = collision_penalty
        super(ProgressReward, self).__init__()
        self._track = track
        self._current_progress = None

    def _compute_progress(self, obs, info):
        assert 'pose' in obs and 'lap_count' in info
        point = obs['pose'][0:2]
        progress = info['lap_count'] + self._track.get_progress(point)
        return progress

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        if info["collision"]:
            reward = - self._collision_penalty
        else:
            reward = 100 * (self._compute_progress(next_state, info) - self._compute_progress(state, info))
        return reward