import math

import gym
import numpy as np
from numba import njit


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
    @njit(fastmath=False, cache=True)
    def _polar2cartesian(dist, angle, n_bins, res):
        occupancy_map = np.zeros(shape=(n_bins, n_bins), dtype=np.uint8)
        xx = dist * np.cos(angle)
        yy = dist * np.sin(angle)
        xi, yi = np.floor(xx / res), np.floor(yy / res)
        for px, py in zip(xi, yi):
            row = min(max(n_bins // 2 + py, 0), n_bins - 1)
            col = min(max(n_bins // 2 + px, 0), n_bins - 1)
            if row < n_bins - 1 and col < n_bins - 1:
                # in this way, then >max_range we dont fill the occupancy map in order to let a visible gap
                occupancy_map[int(row), int(col)] = 255
        return np.expand_dims(occupancy_map, -1)

    def observation(self, observation):
        assert 'scan' in observation
        scan = observation['scan']
        scan_angles = self.sim.agents[0].scan_angles  # assumption: all the lidars are equal in ang. spectrum
        observation['lidar_occupancy'] = self._polar2cartesian(scan, scan_angles, self._n_bins, self._resolution)
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

    def __init__(self, env, obs_list=[]):
        super(FilterObservationWrapper, self).__init__(env)
        self._obs_list = obs_list
        self.observation_space = gym.spaces.Dict({obs: self.env.observation_space[obs] for obs in obs_list})

    def _filter_obs(self, original_obs):
        new_obs = {}
        for obs in self._obs_list:
            assert obs in original_obs
            new_obs[obs] = original_obs[obs]
        return new_obs

    def step(self, action):
        original_obs, reward, done, info = super().step(action)
        new_obs = self._filter_obs(original_obs)
        # add original state into the info
        new_info = info
        new_info['state'] = {name: value for name, value in original_obs.items()}
        return new_obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        new_obs = self._filter_obs(obs)
        return new_obs


class FixSpeedControl(gym.Wrapper):
    """
    reduce original problem to control only the steering angle
    """

    def __init__(self, env, fixed_speed: float = 2.0):
        super(FixSpeedControl, self).__init__(env)
        self._fixed_speed = fixed_speed
        self.action_space = gym.spaces.Dict({'steering': self.env.action_space['steering']})

    def step(self, action):
        new_action = {'steering': action['steering'], 'velocity': self._fixed_speed}
        return super().step(new_action)


class FixResetWrapper(gym.Wrapper):
    """Fix a reset mode to sample initial condition from starting grid or randomly over the track."""

    def __init__(self, env, mode):
        assert mode in ['grid', 'random']
        self._mode = mode
        super(FixResetWrapper, self).__init__(env)

    def reset(self):
        return super(FixResetWrapper, self).reset(mode=self._mode)


class LapLimit(gym.Wrapper):
    """Fix a max nr laps for resetting environment."""

    def __init__(self, env, max_episode_laps):
        self._max_episode_laps = max_episode_laps
        super(LapLimit, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = super(LapLimit, self).step(action)
        assert 'lap_count' in info
        if info['lap_count'] >= self._max_episode_laps:
            done = True
        return obs, reward, done, info


class ElapsedTimeLimit(gym.Wrapper):
    """Fix a max nr laps for resetting environment."""

    def __init__(self, env, max_episode_duration):
        self._max_episode_duration = max_episode_duration
        super(ElapsedTimeLimit, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = super(ElapsedTimeLimit, self).step(action)
        assert 'lap_time' in info
        if info['lap_time'] >= self._max_episode_duration:
            done = True
        return obs, reward, done, info


class FrameSkip(gym.Wrapper):
    def __init__(self, env, frame_skip: int):
        self._frame_skip = frame_skip
        super(FrameSkip, self).__init__(env)

    def step(self, action):
        R = 0
        for t in range(self._frame_skip):
            obs, reward, done, info = self.env.step(action)
            R += reward
            if done:
                break
        return obs, R, done, info


class TerminateOnlyOnTimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps: int):
        self.max_episode_steps = max_episode_steps
        super(TerminateOnlyOnTimeLimit, self).__init__(env)
        self._step = 0

    def reset(self, **kwargs):
        self._step = 0
        return super(TerminateOnlyOnTimeLimit, self).reset(**kwargs)

    def step(self, action):
        self._step += 1
        obs, reward, _, info = super(TerminateOnlyOnTimeLimit, self).step(action)
        done = self._step >= self.max_episode_steps
        return obs, reward, done, info


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


if __name__ == "__main__":
    test_lidar_occupancy_wrapper(render=True)
