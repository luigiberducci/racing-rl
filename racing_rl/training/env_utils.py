import gym
from gym.wrappers import RescaleAction, TimeLimit
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from racing_rl.envs.wrappers import LidarOccupancyObservation, FilterObservationWrapper, FlattenAction, FixSpeedControl, \
    FrameSkip, NormalizeVelocityObservation
from racing_rl.rewards.min_action import MinSteeringReward
from racing_rl.rewards.progress_based import ProgressReward


def make_base_env(name: str):
    env = gym.make(name)
    env = FixSpeedControl(env, fixed_speed=2.0)
    env = MinSteeringReward(env, env.track)
    env = LidarOccupancyObservation(env, max_range=10.0, resolution=0.25)
    env = FilterObservationWrapper(env, obs_list=['lidar_occupancy', 'velocity'])
    env = FlattenAction(env)
    env = RescaleAction(env, a=-1, b=+1)
    env = NormalizeVelocityObservation(env)
    env = FrameSkip(env, frame_skip=4)
    return env