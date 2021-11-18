import math

import gym
import numpy as np


class LidarOccupancyObservation(gym.ObservationWrapper):
    def __init__(self, env, max_range=10.0, resolution=0.08):
        super(LidarOccupancyObservation, self).__init__(env)
        self._max_range = max_range
        self._resolution = resolution
        self._n_bins = math.ceil(2 * self._max_range / self._resolution)
        # extend observation space
        obs_dict = {}
        for k, space in self.observation_space.spaces.items():
            obs_dict[k] = space
        obs_dict['lidar_occupancy'] = gym.spaces.Box(low=0, high=255, dtype=np.uint8,
                                                     shape=(self._n_bins, self._n_bins, 1))
        self.observation_space = gym.spaces.Dict(obs_dict)

    @staticmethod
    def _polar2cartesian(dist, angle):
        x = dist * np.cos(angle)
        y = dist * np.sin(angle)
        return x, y

    def observation(self, observation):
        assert 'scan' in observation
        scan = observation['scan']
        scan_angles = self.sim.agents[0].scan_angles  # assumption: all the lidars are equal in ang. spectrum
        occupancy_map = np.zeros(shape=(self._n_bins, self._n_bins), dtype=np.uint8)
        xx, yy = self._polar2cartesian(scan, scan_angles)
        xi, yi = np.floor(xx / self._resolution), np.floor(yy / self._resolution)
        for px, py in zip(xi, yi):
            row = np.clip(self._n_bins // 2 + py, 0, self._n_bins - 1)
            col = np.clip(self._n_bins // 2 + px, 0, self._n_bins - 1)
            if row < self._n_bins - 1 and col < self._n_bins - 1:
                # in this way, then >max_range we dont fill the occupancy map in order to let a visible gap
                occupancy_map[int(row), int(col)] = 255
        observation['lidar_occupancy'] = np.expand_dims(occupancy_map, axis=-1)
        return observation


class FlattenAction(gym.ActionWrapper):
    """Action wrapper that flattens the action."""

    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)

    def action(self, action):
        return gym.spaces.utils.unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return gym.spaces.utils.flatten(self.env.action_space, action)


class FilterObservationWrapper(gym.Wrapper):
    """
    observation wrapper that filter a single observation and return is without dictionary,
    all the observable quantities are moved to the info as `state`
    """

    def __init__(self, env, obs_name="lidar_occupancy"):
        super(FilterObservationWrapper, self).__init__(env)
        self._obs_name = obs_name
        self.observation_space = self.env.observation_space[obs_name]

    def step(self, action):
        obs, reward, done, info = super().step(action)
        assert self._obs_name in obs
        new_obs = obs[self._obs_name]
        new_info = info
        new_info['state'] = {}
        for name in obs:
            new_info['state'][name] = obs[name]
        return new_obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        assert self._obs_name in obs
        new_obs = obs[self._obs_name]
        return new_obs


"""
class AccelerationControlWrapper(gym.Wrapper):
    def __init__(self, env):
        super(AccelerationControlWrapper, self).__init__(env)
        steering_vel_min, steering_vel_max = self.sim.params['sv_min'], self.sim.params['sv_max']
        acceleration_min, acceleration_max = self.sim.params['sv_min'], self.sim.params['sv_max']
        self.action_space = gym.spaces.Dict({
            'steering_velocity': gym.spaces.Box(low=steering_vel_min, high=steering_vel_max, shape=()),
            'acceleration': gym.spaces.Box(low=acceleration_min, high=acceleration_max, shape=()),
        })
        self._steering = 0.0
        self._velocity = 0.0

    def reset(self, **kwargs):
        self._steering = 0.0
        self._velocity = 0.0
        return super().reset(**kwargs)

    def step(self, action):
        self._steering += self.timestep * action['steering_velocity']
        self._velocity += self.timestep * action['acceleration']
        original_action = {'steering': self._steering,
                           'velocity': self._velocity}
        return super(AccelerationControlWrapper, self).step(original_action)
"""


def test_wrapped_env(env, render):
    for i in range(5):
        print(f"episode {i + 1}")
        obs = env.reset(mode='random')
        for j in range(500):
            obs, reward, done, info = env.step(env.action_space.sample())
            if render:
                env.render()
    # check env
    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env)
        print("[Result] env ok")
    except Exception as ex:
        print("[Result] env not compliant wt openai-gym standard")
        print(ex)


def test_lidar_occupancy_wrapper(render=False):
    from racing_rl.envs.single_agent_env import SingleAgentRaceEnv
    env = SingleAgentRaceEnv("Melbourne")
    env = LidarOccupancyObservation(env, max_range=10)
    test_wrapped_env(env, render)


def test_acceleration_control_wrapper(render=False):
    from racing_rl.envs.single_agent_env import SingleAgentRaceEnv
    env = SingleAgentRaceEnv("Melbourne")
    env = AccelerationControlWrapper(env)
    test_wrapped_env(env, render)


if __name__ == "__main__":
    test_acceleration_control_wrapper(render=True)
