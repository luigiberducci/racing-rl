from typing import Dict, Any

import numpy as np

import f1tenth_gym
import gym

from core.track import Track
from f110_gym.envs import F110Env
from f110_gym.envs.rendering import EnvRenderer


class SingleAgentRaceEnv(F110Env):
    """
    This class implement a base wrapper to the original f1tenth_gym environment tailored on Single-Agent race,
    introducing the following changes:
        - observation and action spaces defined as dictionary with low/high limits of each observable quantities
        - automatic reset of the agent initial position
        - fix rendering issue based on map filepath
    """

    def __init__(self, map_name: str, params: Dict[str, Any] = None, seed: int = 0):
        self._track = Track.from_track_name(map_name)
        sim_params = params if params else self._default_sim_params
        super(SingleAgentRaceEnv, self).__init__(map=self._track.filepath, map_ext=self._track.ext,
                                                 params=sim_params, num_agents=1, seed=seed)
        self._scan_size = self.sim.agents[0].scan_simulator.num_beams
        self._scan_range = self.sim.agents[0].scan_simulator.max_range

    @property
    def observation_space(self):
        """
            scan: lidar data (m)
            pose: x, y, z coordinate (m)
            velocity: linear x velocity (m/s), linear y velocity (m/s), angular velocity (rad/s)
            collision: indicator if the agent is in collision (bool)
        """
        return gym.spaces.Dict({
            'scan': gym.spaces.Box(low=0.0, high=self._scan_range, shape=(self._scan_size,)),
            'pose': gym.spaces.Box(low=np.NINF, high=np.PINF, shape=(3,)),
            'velocity': gym.spaces.Box(low=np.NINF, high=np.PINF, shape=(3,)),
            'collision': gym.spaces.Discrete(2),
        })

    @property
    def action_space(self):
        """
            steering: desired steering angle (rad)
            velocity: desired velocity (m/s)
        """
        steering_low, steering_high = self.sim.params['s_min'], self.sim.params['s_max']
        velocity_low, velocity_high = self.sim.params['v_min'], self.sim.params['v_max']
        return gym.spaces.Dict({
            "steering": gym.spaces.Box(low=steering_low, high=steering_high, shape=()),
            "velocity": gym.spaces.Box(low=velocity_low, high=velocity_high, shape=())
        })

    @property
    def _default_sim_params(self):
        return {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74,
                'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319,
                'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

    @staticmethod
    def _get_flat_action(action: Dict[str, float]):
        assert 'steering' in action and 'velocity' in action
        flat_action = np.array([[action['steering'], action['velocity']]])
        assert flat_action.shape == (1, 2), f'the actions dict-array conversion returns wrong shape {flat_action.shape}'
        return flat_action

    def _prepare_obs(self, old_obs):
        assert all([f in old_obs for f in ['scans', 'poses_x', 'poses_y', 'poses_theta',
                                           'linear_vels_x', 'linear_vels_y', 'ang_vels_z',
                                           'collisions']]), f'obs keys are {old_obs.keys()}'
        # Note: the original env returns `scan` values > `max_range`. To keep compatibility wt obs-space, we clip it
        obs = {'scan': np.clip(old_obs['scans'][0], 0, self._scan_range),
               'pose': np.array([old_obs['poses_x'][0], old_obs['poses_y'][0], old_obs['poses_theta'][0]]),
               'velocity': np.array(
                   [old_obs['linear_vels_x'][0], old_obs['linear_vels_y'][0], old_obs['ang_vels_z'][0]]),
               'collision': 1 if old_obs['collisions'] else 0}
        return obs

    def _prepare_info(self, old_obs, old_info):
        assert all([f in old_obs for f in ['lap_times', 'lap_counts']]), f'obs keys are {old_obs.keys()}'
        assert all([f in old_info for f in ['checkpoint_done']]), f'info keys are {old_info.keys()}'
        info = {'checkpoint_done': old_info['checkpoint_done'][0],
                'lap_time': old_obs['lap_times'][0],
                'lap_count': old_obs['lap_counts'][0],
                }
        return info

    def step(self, action):
        """
        Note: `step` is used in the `reset` method of the original environment, to initialize the simulators
        For this reason, we cannot rid off the original fields in the observation completely.
        We overcome this by checking if the action is an array (then called in the reset) or a dictionary (otherwise)
        """
        if type(action) == np.ndarray:
            obs, reward, done, info = super().step(action)
        else:
            flat_action = self._get_flat_action(action)
            original_obs, reward, done, original_info = super().step(flat_action)
            obs = self._prepare_obs(original_obs)
            info = self._prepare_info(original_obs, original_info)
            done = bool(done)
        return obs, reward, done, info

    def reset(self, mode: str = 'grid'):
        """ reset modes:
                - grid: reset the agent position on the first waypoint
                - random: reset the agent position on a random waypoint
        """
        assert mode in ['grid', 'random']
        if mode == "grid":
            waypoint_id = 0
        elif mode == "random":
            waypoint_id = np.random.randint(self._track.centerline.shape[0] - 1)
        else:
            raise NotImplementedError(f"reset mode {mode} not implemented")
        # compute x, y, theta
        assert 0 <= waypoint_id < self._track.centerline.shape[0] - 1
        wp, next_wp = self._track.centerline[waypoint_id], self._track.centerline[waypoint_id + 1]
        theta = np.arctan2(wp[1] - next_wp[1], wp[0] - next_wp[0])
        pose = [wp[0], wp[1], theta]
        # call original method
        original_obs, reward, done, original_info = super().reset(poses=np.array([pose]))
        obs = self._prepare_obs(original_obs)
        return obs

    def render(self, mode='human'):
        WINDOW_W, WINDOW_H = 1000, 800
        if self.renderer is None:
            # first call, initialize everything
            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            F110Env.renderer.update_map(self._track.filepath, self._track.ext)
        super(SingleAgentRaceEnv, self).render(mode)


if __name__ == "__main__":
    env = SingleAgentRaceEnv("Catalunya")
    for i in range(3):
        print(f"episode {i + 1}")
        obs = env.reset(mode='random')
        for j in range(500):
            obs, reward, done, info = env.step({'steering': 0.0, 'velocity': 2.0})
            env.render()
        input("next?")
    # check env
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
