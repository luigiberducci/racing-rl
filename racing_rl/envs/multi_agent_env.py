import collections
import math
from typing import Dict, Any, List, Callable

import numpy as np

import gym

from racing_rl.envs.track import Track
from f110_gym.envs import F110Env
from f110_gym.envs.rendering import EnvRenderer


def init_basic_ctrl(track, n_npcs, fixed_speed, wheelbase):
    """ return a basic pure-pursuit controller """
    from racing_rl.baseline.PurePursuitPlanner import PurePursuitPlanner
    pp = PurePursuitPlanner(track, wb=wheelbase, fixed_speed=fixed_speed)
    # note: the PurePursuitPlanner return speed, steering, while we expect steering, speed
    # then the returned action is flipped with [::-1]
    return [pp for _ in range(n_npcs)]


def compute_sampling_params(track):
    approx_len = 0.0
    for wp, nwp in zip(track.centerline[:-1, :], track.centerline[1:, :]):
        x_diff = nwp[0] - wp[0]
        y_diff = nwp[1] - wp[1]
        approx_len += np.linalg.norm([y_diff, x_diff])  # approx distance by summing linear segments
    if approx_len < 15.0:  # small tracks, with less than 10m centerline
        return 1.5, 3.0
    else:
        return 3.0, 10.0


class MultiAgentRaceEnv(F110Env):
    """
    This class implement a base wrapper to the original f1tenth_gym environment for SingleAgent in presence of other cars,
    introducing the following changes:
        - observation and action spaces defined as dictionary with low/high limits of each observable quantities
        - automatic reset of the agents initial position
        - automatic control of NPC according to any input controller,
        - fix rendering issue based on map filepath
    """

    def __init__(self, map_name: str, gui: bool = False, n_npcs: int = 1, npc_controllers: List[Callable] = None,
                 params: Dict[str, Any] = None, term_overcoming: bool = True, seed: int = None):
        # sim params
        self._track = Track.from_track_name(map_name)
        seed = np.random.randint(0, 1000000) if seed is None else seed
        sim_params = params if params else self._default_sim_params
        self._velocity_low, self._velocity_high = 1.0, 2.5  # 0 velocity could cause division-by-zero
        # multi-agent configuration
        self._n_npcs = n_npcs
        self._agent_ids = ["ego"] + [f"npc{i}" for i in range(self._n_npcs)]
        wheelbase = sim_params['lf'] + sim_params['lr']
        self._npc_controllers = init_basic_ctrl(self._track, self._n_npcs, fixed_speed=1.5,
                                                wheelbase=wheelbase) if npc_controllers is None else npc_controllers
        # sampling intial conditions
        self._min_dist, self._max_dist = compute_sampling_params(self._track)
        #
        super(MultiAgentRaceEnv, self).__init__(map=self._track.filepath, map_ext=self._track.ext,
                                                params=sim_params, num_agents=len(self._agent_ids), seed=seed)
        self.add_render_callback(render_callback)
        self._scan_size = self.sim.agents[0].scan_simulator.num_beams
        self._scan_range = self.sim.agents[0].scan_simulator.max_range
        # rendering
        self._gui = gui
        self._render_freq = 10
        self._step = 0
        # keep state for playing npcs
        self._complete_state = None
        # termination conditions
        self._terminate_on_overcoming = term_overcoming

    @property
    def track(self):
        return self._track

    @property
    def observation_space(self):
        """
        The observation space is refered to the ego frame
            scan: lidar data (m)
            pose: x, y, z coordinate (m)
            velocity: linear x velocity (m/s), linear y velocity (m/s), angular velocity (rad/s)
        """
        return gym.spaces.Dict({
            'scan': gym.spaces.Box(low=0.0, high=self._scan_range, shape=(self._scan_size,)),
            'pose': gym.spaces.Box(low=np.NINF, high=np.PINF, shape=(3,)),
            'velocity': gym.spaces.Box(low=-5, high=20, shape=(1,)),
            'collision': gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
        })

    @property
    def action_space(self):
        """
            steering: desired steering angle (rad)
            velocity: desired velocity (m/s)
        """
        steering_low, steering_high = self.sim.params['s_min'], self.sim.params['s_max']
        return gym.spaces.Dict({
            "steering": gym.spaces.Box(low=steering_low, high=steering_high, shape=()),
            "velocity": gym.spaces.Box(low=self._velocity_low, high=self._velocity_high, shape=())
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

    def _extract_agent_state(self, obs, agent_id):
        assert agent_id in self._agent_ids
        n = self._agent_ids.index(agent_id)
        state = {
            'scan': np.clip(obs['scans'][n], 0, self._scan_range),
            'pose': np.array([obs['poses_x'][n], obs['poses_y'][n], obs['poses_theta'][n]]),
            'velocity': np.array([obs['linear_vels_x'][n]]),
            'collision': 1.0 if obs['collisions'][n] else 0.0}
        return state

    def _prepare_obs(self, old_obs):
        assert all([f in old_obs for f in ['scans', 'poses_x', 'poses_y', 'poses_theta',
                                           'linear_vels_x', 'linear_vels_y', 'ang_vels_z',
                                           'collisions']]), f'obs keys are {old_obs.keys()}'
        # Note: the original env returns `scan` values > `max_range`. To keep compatibility wt obs-space, we cap it
        # store complete state
        self._complete_state = {agent_id: self._extract_agent_state(old_obs, agent_id) for agent_id in self._agent_ids}
        # observe only ego state
        obs = collections.OrderedDict(self._complete_state["ego"])
        return obs

    def _prepare_info(self, old_obs, action, old_info):
        assert all([f in old_obs for f in
                    ['lap_times', 'lap_counts', 'collisions', 'linear_vels_x']]), f'obs keys are {old_obs.keys()}'
        assert all([f in action for f in ['steering', 'velocity']]), f'action keys are {action.keys()}'
        assert all([f in old_info for f in ['checkpoint_done']]), f'info keys are {old_info.keys()}'
        info = {'checkpoint_done': old_info['checkpoint_done'][0],
                'lap_time': old_obs['lap_times'][0],
                'lap_count': old_obs['lap_counts'][0],
                'collision': old_obs['collisions'][0],
                'velocity': old_obs['linear_vels_x'][0],
                'action': action
                }
        # store the state of other cars in the info dictionary
        info.update({npc: self._complete_state[npc] for npc in self._agent_ids[1:]})
        return info

    def _prepare_multiagent_action(self, action):
        npc_actions = []
        for i, (agent, ctrl) in enumerate(zip(self._agent_ids[1:], self._npc_controllers)):
            obs = self._complete_state[agent]
            act = list(self._npc_controllers[i].predict(obs).values())
            npc_actions.append(act)
        multiagent_action = np.concatenate([action, np.array(npc_actions)])
        return multiagent_action

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
            multiagent_action = self._prepare_multiagent_action(flat_action)
            original_obs, reward, done, original_info = super().step(multiagent_action)
            obs = self._prepare_obs(original_obs)
            info = self._prepare_info(original_obs, action, original_info)
            done = bool(done)
        # rendering
        if self._gui and self._step % self._render_freq == 0:
            self.render()
        self._step += 1
        # termination condition
        if self._terminate_on_overcoming and self._complete_state is not None:
            for car, state in self._complete_state.items():
                position = state["pose"][:2]
                done = done or (self.track.get_progress(position) > 1.0)
        return obs, reward, done, info

    def _compute_poses(self, ego_wp, min_dist, max_dist):
        """ Iteratively sample the next agent to be within [min_dist, max_dist] from the previous agent."""
        track_len = self._track.centerline.shape[0]
        poses = []
        current_wp_id = ego_wp
        while len(poses) < self.num_agents:
            # compute pose from current wp_id
            wp, next_wp = self._track.centerline[current_wp_id], self._track.centerline[current_wp_id + 1]
            theta = np.arctan2(next_wp[1] - wp[1], next_wp[0] - wp[0])
            pose = [wp[0], wp[1], theta]
            poses.append(pose)
            # find id of next waypoint which has mind <= dist <= maxd
            first_id, interval_len = None, None   # first wp id with dist > mind, len of the interval btw first/last wp
            pnt_id = current_wp_id  # moving pointer to scan the next waypoints
            dist = 0.0
            while dist < max_dist:
                # sanity check
                if pnt_id > track_len - 1:
                    pnt_id = 0
                # increment distance
                x_diff = self._track.centerline[pnt_id][0] - self._track.centerline[pnt_id - 1][0]
                y_diff = self._track.centerline[pnt_id][1] - self._track.centerline[pnt_id - 1][1]
                dist = dist + np.linalg.norm([y_diff, x_diff])  # approx distance by summing linear segments
                # look for sampling interval
                if first_id is None and dist >= min_dist:   # not found first id yet
                    first_id = pnt_id
                    interval_len = 0
                if first_id is not None and dist <= max_dist:  # found first id, increment interval length
                    interval_len += 1
                pnt_id += 1
            # sample next waypoint
            current_wp_id = first_id + np.random.randint(0, interval_len) % (track_len - 1)
        return poses

    def reset(self, mode: str = 'grid'):
        """ reset modes:
                - grid: reset the agent position on the first waypoint
                - random: reset the agent position on a random waypoint
        """
        # reset npc controllers
        rnd_in = lambda m, M: m + (M - m) * np.random.random()
        [ctrl.reset(fixed_speed=rnd_in(self._velocity_low+0.1, self._velocity_high-0.1)) for ctrl in self._npc_controllers]
        # sample initial conditions
        assert mode in ['grid', 'random']
        if mode == "grid":
            waypoint_id = 0
        elif mode == "random":
            waypoint_id = np.random.randint(self._track.centerline.shape[0] // 2)
        else:
            raise NotImplementedError(f"reset mode {mode} not implemented")
        poses = self._compute_poses(waypoint_id, self._min_dist, self._max_dist)
        # call original method
        original_obs, reward, done, original_info = super().reset(poses=np.array(poses))
        obs = self._prepare_obs(original_obs)
        self._step = 0
        return obs

    def render(self, mode='human'):
        WINDOW_W, WINDOW_H = 1000, 800
        if self.renderer is None:
            # first call, initialize everything
            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            F110Env.renderer.update_map(self._track.filepath, self._track.ext)
        super(MultiAgentRaceEnv, self).render(mode)


def render_callback(env_renderer):
    # custom extra drawing function
    e = env_renderer

    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 800
    e.right = right + 800
    e.top = top + 800
    e.bottom = bottom - 800


if __name__ == "__main__":
    env = MultiAgentRaceEnv("Fallstudien", n_npcs=2)
    for i in range(3):
        print(f"episode {i + 1}")
        obs = env.reset(mode='random')
        for j in range(500):
            obs, reward, done, info = env.step({'steering': 0.0, 'velocity': 1.5})
            print(info["collision"])
            env.render()
            if done:
                break
    # check env
    try:
        from stable_baselines3.common.env_checker import check_env

        check_env(env)
        print("[Result] env ok")
    except Exception as ex:
        print("[Result] env not compliant wt openai-gym standard")
        print(ex)
