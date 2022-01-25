from typing import Dict, Any

import gym
from gym.wrappers import RescaleAction, FlattenObservation
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from racing_rl.envs.wrappers import LidarOccupancyObservation, FilterObservationWrapper, FlattenAction, FixSpeedControl, \
    FrameSkip, NormalizeVelocityObservation, FrameStackOnChannel
from racing_rl.rewards.core.reward import RewardWrapper
from racing_rl.rewards.hrs_potential.multiagent_potential import HRSConservativeReward
from racing_rl.rewards.progress.min_action import MinActionReward
from racing_rl.rewards.progress.progress_based import ProgressReward


def get_reward_wrapper(reward: str, coll_penalty: float = 0.0):
    if reward == 'min_action':
        return lambda env: RewardWrapper(env, reward_fn=MinActionReward(action_space=env.action_space, collision_penalty=coll_penalty))
    elif reward == 'progress':
        return lambda env: RewardWrapper(env, reward_fn=ProgressReward(track=env.track, collision_penalty=coll_penalty))
    elif reward == 'only_progress':
        return lambda env: RewardWrapper(env, reward_fn=ProgressReward(track=env.track, collision_penalty=0.0))
    elif reward == 'hrs_conservative':
        return lambda env: RewardWrapper(env, reward_fn=HRSConservativeReward(track=env.track, action_space=env.action_space))
    raise NotImplementedError(f'reward {reward} not implemented')


def make_base_env(name: str, reward: str, collision_penalty: float, only_steering: bool,
                  include_velocity: bool, frame_aggregation: str = None) -> (gym.Env, Dict[str, Any]):
    env_params = {
        'name': name,
        'reward': {'name': reward, 'collision_penalty': collision_penalty},
        'actions': {'only_steering': only_steering, 'fixed_speed': 2.0, 'frame_skip': 4},
        'observations': {'max_range': 10.0, 'resolution': 0.25, 'include_velocity': include_velocity,
                         'frame_aggregation': frame_aggregation, 'n_frame_aggregated': 2}
    }
    env = gym.make(name)
    # define action space
    if only_steering:
        env = FixSpeedControl(env, fixed_speed=env_params['actions']['fixed_speed'])
    # define reward
    env = get_reward_wrapper(env_params['reward']['name'], env_params['reward']['collision_penalty'])(env)
    # define observation space
    env = LidarOccupancyObservation(env, max_range=env_params['observations']['max_range'],
                                    resolution=env_params['observations']['resolution'])
    if include_velocity:
        env = FilterObservationWrapper(env, obs_list=['lidar_occupancy', 'velocity'])
        env = NormalizeVelocityObservation(env)
        env = FrameSkip(env, skip=env_params['actions']['frame_skip'])
    else:
        assert frame_aggregation is not None, "if not obs velocity, then expected frame aggregation (max,stack)"
        env = FilterObservationWrapper(env, obs_list=['lidar_occupancy'])
        env = FlattenObservation(env)
        if frame_aggregation == "max":
            env = MaxAndSkipEnv(env, skip=env_params['actions']['frame_skip'])
        elif frame_aggregation == "stack":
            env = FrameStackOnChannel(env, num_stack=env_params['observations']['n_frame_aggregated'])
            env = FrameSkip(env, skip=env_params['actions']['frame_skip'])
        else:
            raise NotImplementedError(f"frame aggregation {frame_aggregation} is not defined")
    env = FlattenAction(env)
    env = RescaleAction(env, a=-1, b=+1)
    return env, env_params
