import collections
import math

import gym
import numpy as np
from gym.spaces import Box
from gym.wrappers import LazyFrames
from numba import njit


class LidarOccupancyObservation(gym.ObservationWrapper):
    def __init__(self, env, max_range: float = 10.0, resolution: float = 0.08, degree_fow: int = 270):
        super(LidarOccupancyObservation, self).__init__(env)
        self._max_range = max_range
        self._resolution = resolution
        self._degree_fow = degree_fow
        self._n_bins = math.ceil(2 * self._max_range / self._resolution)
        # extend observation space
        obs_dict = collections.OrderedDict()
        for k, space in self.observation_space.spaces.items():
            obs_dict[k] = space
        obs_dict['lidar_occupancy'] = gym.spaces.Box(low=0, high=255, dtype=np.uint8,
                                                     shape=(1, self._n_bins, self._n_bins))
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
        return np.expand_dims(occupancy_map, 0)

    def observation(self, observation):
        assert 'scan' in observation
        scan = observation['scan']
        scan_angles = self.sim.agents[0].scan_angles  # assumption: all the lidars are equal in ang. spectrum
        # reduce fow
        mask = abs(scan_angles) <= np.deg2rad(self._degree_fow / 2.0)  # 1 for angles in fow, 0 for others
        scan = np.where(mask, scan, np.Inf)
        observation['lidar_occupancy'] = self._polar2cartesian(scan, scan_angles, self._n_bins, self._resolution)
        return observation


class PreCommandObservation(gym.Wrapper):
    """
    Stack the last command to the observation.
    """
    def __init__(self, env):
        super(PreCommandObservation, self).__init__(env)
        # extend observation space
        assert isinstance(env.action_space, gym.spaces.Dict)
        obs_dict = collections.OrderedDict()
        for k, space in self.observation_space.spaces.items():
            obs_dict[k] = space
        self._last_action = None
        for action in env.action_space.spaces.keys():
            act_space = env.action_space[action]
            obs_dict[f"last_{action}"] = gym.spaces.Box(low=np.array([act_space.low]), high=np.array([act_space.high]), shape=(1,))
        self.observation_space = gym.spaces.Dict(obs_dict)

    def reset(self, **kwargs):
        self._last_action = {act: np.array([0.0]) for act in self.action_space.spaces.keys()}
        obs = super(PreCommandObservation, self).reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        self._last_action = action
        obs, reward, done, info = super(PreCommandObservation, self).step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def observation(self, observation):
        for action in self.action_space.spaces.keys():
            observation[f"last_{action}"] = self._last_action[action].reshape(1,)
        return observation


class FrameStackOnChannel(gym.Wrapper):
    r"""
    Observation wrapper that stacks the observations in a rolling manner.

    Implementation from gym.wrappers but squeeze observation (then removing channel dimension),
    in order to stack over the channel dimension.
    """

    def __init__(self, env, num_stack, lz4_compress=False):
        super(FrameStackOnChannel, self).__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = collections.deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def _get_observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(np.squeeze(observation))  # assume 1d channel dimension and remove it
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self._get_observation()


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
        new_obs = collections.OrderedDict()
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


class FixSpeedControl(gym.ActionWrapper):
    """
    reduce original problem to control only the steering angle
    """

    def __init__(self, env, fixed_speed: float = 2.0):
        super(FixSpeedControl, self).__init__(env)
        self._fixed_speed = fixed_speed
        self.action_space = gym.spaces.Dict({'steering': self.env.action_space['steering']})

    def action(self, action):
        assert 'steering' in action
        new_action = {'steering': action['steering'], 'velocity': self._fixed_speed}
        return new_action


class ConstrainedSpeedControl(gym.ActionWrapper):
    """
    control the steering angle and a constrained increment of the velocity
    """

    def __init__(self, env, max_increment: float = 1.0, max_decrement: float = -1.0):
        super(ConstrainedSpeedControl, self).__init__(env)
        assert max_increment > 0 and max_decrement < 0, f'not valid arguments: max_increment {max_increment}, max_decrement {max_decrement}'
        self._max_increment = max_increment
        self._max_decrement = max_decrement
        self._last_velocity = 0.0
        self._old_action_space = self.action_space
        self.action_space = gym.spaces.Dict({'steering': self.env.action_space['steering'],
                                             'delta_velocity': gym.spaces.Box(low=self._max_decrement,
                                                                              high=self._max_increment, shape=())})

    def reset(self, **kwargs):
        self._last_velocity = 0.0
        return super(ConstrainedSpeedControl, self).reset(**kwargs)

    def action(self, action):
        assert action in self.action_space
        velocity_low = self._old_action_space['velocity'].low
        velocity_high = self._old_action_space['velocity'].high
        velocity = np.clip(self._last_velocity + action['delta_velocity'], velocity_low, velocity_high)
        new_action = {'steering': action['steering'], 'velocity': velocity}
        return new_action


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


class NormalizeVelocityObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        assert 'velocity' in obs
        low = self.observation_space['velocity'].low
        high = self.observation_space['velocity'].high
        obs['velocity'] = 2 * ((obs['velocity'] - low) / (high - low)) - 1
        return obs


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
    def __init__(self, env, skip: int):
        self._frame_skip = skip
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
