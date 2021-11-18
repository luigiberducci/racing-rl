import gym
import torch as th
import torch.nn as nn
from gym.wrappers import FrameStack

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from racing_rl.envs.wrappers import LidarOccupancyObservation, FilterObservationWrapper, FlattenAction
from racing_rl.rewards.progress_based import ProgressReward


env = gym.make("SingleAgentMelbourne-v0")
env = ProgressReward(env, env.track)
env = LidarOccupancyObservation(env, max_range=10.0, resolution=0.078125)   # magic number to have img 256x256
env = FilterObservationWrapper(env, obs_name='lidar_occupancy')
env = FlattenAction(env)
env = gym.wrappers.RescaleAction(env, a=-1, b=+1)
env = FrameStack(env, num_stack=4)

model = PPO("CnnPolicy", env, verbose=1)
model.learn(1000)