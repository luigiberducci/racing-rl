import gym
import numpy as np

from racing_rl.envs.track import Track
from racing_rl.rewards.core.reward import RewardFunction


class HRSConservativeReward(RewardFunction):
    """
    task specification :=
        achieve (complete 1 lap)
        ensuring (no collision with walls or cars)
        encouraging (follow with safety distance)
        encouraging (smooth steering)
    """

    def _clip_and_norm(self, v, min, max):
        return (np.clip(v, min, max) - min) / (max-min)

    def __init__(self, track, observation_space):
        self._track = track
        self._observation_space = observation_space
        super(HRSConservativeReward, self).__init__()

    def _target_potential(self, state, info):
        """ target requirement := achieve (complete 1 lap) """
        assert 'pose' in state and 'lap_count' in info
        position = state['pose'][0:2]
        progress = info['lap_count'] + self._track.get_progress(position)
        return progress

    def _safety_potential(self, state, info):
        """ safety requirement := ensure (no collision) """
        if info["collision"]:
            return 0.0
        return 1.0

    def _comfortable_steering(self, state, info):
        """ comfort requirement := encourage (steering smoothness)"""
        assert 'last_steering' in state and 'last_velocity' in state
        maxsteering = self._observation_space["last_steering"].high[0]
        reward = 1 - (self._clip_and_norm(state["last_steering"]**2, 0.0, maxsteering**2))
        return reward

    def _limit_velocity(self, action, info):
        """
        comfort requirement := encourage (mv <= v <= Mv) =
        encourage (mv <= v) and encourage (v <= Mv)
        """
        minv, maxv = 1.5, 2.0
        minv_reward = self._clip_and_norm(action["velocity"], 0.0, minv)
        maxv_reward = 1 - self._clip_and_norm(action["velocity"], maxv, 5.0)
        return 0.5 * (minv_reward + maxv_reward)

    def _safe_distance(self, state, info):
        """ requirement := if leading car then keep a safety distance """
        # params
        target_distance = -2.0
        # compute distance
        ego_position = self._track.get_progress(state["pose"][0:2], return_meters=True)
        npc_position = self._track.get_progress(info["npc0"]["pose"][0:2], return_meters=True)
        reward = self._clip_and_norm((ego_position - npc_position) ** 2, 0.0,  target_distance**2)   # dist -> 0, r -> 0
        return reward

    def _comfort_diff_potential(self, state, action, next_state, info):
        # hierarchical weights
        safety_w, target_w = self._safety_potential(state, info), self._target_potential(state, info)
        n_safety_w, n_target_w = self._safety_potential(next_state, info), self._target_potential(next_state, info)
        w, nw = safety_w * target_w, n_safety_w * n_target_w
        # comfort potentials
        safe_dist_reward = nw * self._safe_distance(next_state, info) - w * self._safe_distance(state, info)
        # comfort actions
        steering_reward = nw * self._comfortable_steering(next_state, info) - w * self._comfortable_steering(state, info)
        return steering_reward + safe_dist_reward

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        safety_reward = self._safety_potential(next_state, info) - self._safety_potential(state, info)
        target_reward = self._target_potential(next_state, info) - self._target_potential(state, info)
        comfort_reward = self._comfort_diff_potential(state, action, next_state, info)
        return safety_reward + target_reward + comfort_reward
