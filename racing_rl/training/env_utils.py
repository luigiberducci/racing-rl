import gym
from gym.wrappers import RescaleAction, TimeLimit
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from racing_rl.envs.wrappers import LidarOccupancyObservation, FilterObservationWrapper, FlattenAction
from racing_rl.rewards.progress_based import ProgressReward


def make_base_env(name: str):
    env = gym.make(name)
    env = ProgressReward(env, env.track)
    env = LidarOccupancyObservation(env, max_range=10.0, resolution=0.1)
    env = FilterObservationWrapper(env, obs_name='lidar_occupancy')
    env = FlattenAction(env)
    env = RescaleAction(env, a=-1, b=+1)
    env = MaxAndSkipEnv(env, skip=4)
    return env