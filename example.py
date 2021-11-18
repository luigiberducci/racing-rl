import gym
from gym.wrappers import FrameStack
from stable_baselines3.common.env_checker import check_env

import racing_rl
from racing_rl.envs.wrappers import LidarOccupancyObservation, FlattenAction, FilterObservationWrapper
from racing_rl.rewards.progress_based import ProgressReward

env = gym.make("SingleAgentMelbourne-v0")
env = ProgressReward(env, env.track)
env = LidarOccupancyObservation(env)
env = FilterObservationWrapper(env, obs_name='lidar_occupancy')
env = FlattenAction(env)
env = gym.wrappers.RescaleAction(env, a=-1, b=+1)
env = FrameStack(env, num_stack=5)


for i in range(5):
    obs = env.reset(mode='grid')
    done = False
    t = 0
    while t<1000 and not done:
        t += 1
        obs, reward, done, info = env.step([0.0, 0.5])
        print(reward)
        if info['collision']:
            print("COLLISION")
        env.render()