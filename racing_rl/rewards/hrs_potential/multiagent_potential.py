import gym
import numpy as np

from racing_rl.envs.track import Track
from racing_rl.rewards.core.reward import RewardFunction


class HRSConservativeReward(RewardFunction):
    """
    task specification :=
        achieve (complete 1 lap)
        ensuring (no collision with walls or cars)
        encouraging (dist(ego, leadingcar) <= safety distance)  # negative distance means that ego is behind
        encouraging (dist to right lane <= tolerance)
        encouraging (minv <= velocity <= maxv)
    """

    def __init__(self, track, action_space):
        self._track = track
        self._action_space = action_space
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

    def _comfortable_steering(self, action, info):
        """ comfort requirement := encourage (steering smoothness)"""
        def normalize_action(name, value):
            # normalize action in +-1
            low, high = self._action_space[name].low, self._action_space[name].high
            norm_action = 2 * (value - low) / (high - low) - 1
            return norm_action

        steering = normalize_action("steering", action["steering"])
        reward = 1 - (np.linalg.norm(steering) ** 2)
        return reward

    def _limit_velocity(self, action, info):
        """ comfort requirement := encourage (mv <= v <= Mv)"""
        minv, maxv = 1.75, 2.25
        norm_velocity = (np.clip(action["velocity"], minv, maxv) - minv) / (maxv - minv)    # norm 0,1
        return 1 - (abs(0.5 - norm_velocity) / 0.5) ** 2

    def _safe_distance(self, state, info):
        """ requirement := if leading car then keep a safety distance """
        ego_position = self._track.get_progress(state["pose"][0:2], return_meters=True)
        npc_position = self._track.get_progress(info["npc0"]["pose"][0:2], return_meters=True)
        dist = np.clip(ego_position - npc_position, -100.0, 100.0)
        unsafe_distance, target_distance = -1.0, -2.0
        if dist < 0:    # ego behind
            if dist > -1.0:
                return - 1.0
            elif target_distance - 0.5 <= dist <= target_distance + 0.5:
                return + 1.0
            else:
                return 1.0 - (dist - target_distance) ** 2
        else:
            return 0.0  # if the ego is not behind, the evaluation is not defined

    def _small_crosstrack_error(self, state, info):
        ego_position = state["pose"][0:2]
        idx = self._track.get_id_closest_point2centerline(ego_position)
        closest_point = self._track.centerline[idx][0:2]
        return 1 - (np.linalg.norm([ego_position[0]-closest_point[0], ego_position[1]-closest_point[1]]) ** 2)

    def _comfort_diff_potential(self, state, action, next_state, info):
        # hierarchical weights
        safety_w, target_w = self._safety_potential(state, info), self._target_potential(state, info)
        n_safety_w, n_target_w = self._safety_potential(next_state, info), self._target_potential(next_state, info)
        w, nw = safety_w * target_w, n_safety_w * n_target_w
        # comfort potentials
        safe_dist_reward = nw * self._safe_distance(next_state, info) - w * self._safe_distance(state, info)
        crosstrack_reward = nw * self._small_crosstrack_error(next_state, info) - w * self._safe_distance(state, info)
        # comfort actions
        steering_reward = nw * self._comfortable_steering(action, info)
        velocity_reward = nw * self._limit_velocity(action, info)
        #
        return safe_dist_reward + crosstrack_reward + steering_reward + velocity_reward

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        safety_reward = self._safety_potential(next_state, info) - self._safety_potential(state, info)
        target_reward = self._target_potential(next_state, info) - self._target_potential(state, info)
        comfort_reward = self._comfort_diff_potential(state, action, next_state, info)
        return safety_reward + target_reward + comfort_reward


class HRSAggressivePolicy(gym.Wrapper):
    """
    task specification :=
        achieve (complete 1 lap)
        ensuring (no collision with walls or cars)
        encouraging (dist(ego, leadingcar) > carlength)     # positive distance means that ego is in front
        encouraging (dist to right lane <= tolerance)
        encouraging (minv <= velocity <= maxv)
    """

    def __init__(self, env, track: Track):
        super(HRSAggressivePolicy, self).__init__(env)
        self._track = track

    def reset(self, **kwargs):
        obs = super(HRSAggressivePolicy, self).reset(**kwargs)
        return obs

    def step(self, action):
        obs, _, done, info = super(HRSAggressivePolicy, self).step(action)
        reward = 0.0
        return obs, reward, done, info
