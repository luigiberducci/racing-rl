import math

import gym
import numpy as np


class LidarOccupancyObservation(gym.Wrapper):
    def __init__(self, env, max_range=30.0, resolution=0.5):
        super(LidarOccupancyObservation, self).__init__(env)
        self._max_range = max_range
        self._resolution = resolution
        self._n_bins = math.ceil(2 * self._max_range / self._resolution)
        # extend observation space
        obs_dict = {}
        for k, space in self.observation_space.spaces.items():
            obs_dict[k] = space
        obs_dict['lidar_occupancy'] = gym.spaces.Box(low=0, high=255, dtype=np.uint8, shape=(self._n_bins, self._n_bins, 1))
        self.observation_space = gym.spaces.Dict(obs_dict)

    def _polar2cartesian(self, dist, angle):
        x = dist * np.cos(angle)
        y = dist * np.sin(angle)
        return x, y

    def _augment_obs(self, obs):
        assert 'scan' in obs
        scan = obs['scan']
        scan_angles = self.sim.agents[0].scan_angles  # assumption: all the lidars are equal in ang. spectrum
        occupancy_map = np.zeros(shape=(self._n_bins, self._n_bins), dtype=np.uint8)
        xx, yy = self._polar2cartesian(scan, scan_angles)
        xi, yi = np.floor(xx / self._resolution), np.floor(yy / self._resolution)
        for px, py in zip(xi, yi):
            row = np.clip(self._n_bins // 2 + py, 0, self._n_bins - 1)
            col = np.clip(self._n_bins // 2 + px, 0, self._n_bins - 1)
            if row < self._n_bins-1 and col<self._n_bins-1:
                # in this way, then >max_range we dont fill the occupancy map in order to let a visible gap
                occupancy_map[int(row), int(col)] = 255
        obs['lidar_occupancy'] = np.expand_dims(occupancy_map, axis=-1)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = self._augment_obs(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        obs = self._augment_obs(obs)
        return obs


if __name__ == "__main__":
    from racing_rl.envs.single_agent_env import SingleAgentRaceEnv

    env = SingleAgentRaceEnv("Melbourne")
    env = LidarOccupancyObservation(env, max_range=10)
    for i in range(3):
        print(f"episode {i + 1}")
        obs = env.reset(mode='random')
        for j in range(10):
            obs, reward, done, info = env.step({'steering': 0.0, 'velocity': 5.0})
    # check env
    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env)
        print("[Result] env ok")
    except Exception as ex:
        print("[Result] env not compliant wt openai-gym standard")
        print(ex)
