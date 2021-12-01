import gym
from gym.wrappers import RescaleAction, TimeLimit
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from racing_rl.envs.wrappers import LidarOccupancyObservation, FilterObservationWrapper, FlattenAction, FixSpeedControl, \
    FrameSkip, NormalizeVelocityObservation
from racing_rl.rewards.min_action import MinActionReward
from racing_rl.rewards.progress_based import ProgressReward


def get_reward_wrapper(reward: str, collision_penalty: float = 0.0):
    if reward == 'min_action':
        return lambda env: MinActionReward(env, collision_penalty=collision_penalty)
    elif reward == 'progress':
        return lambda env: ProgressReward(env, env.track, collision_penalty=collision_penalty)
    elif reward == 'only_progress':
        return lambda env: ProgressReward(env, env.track, collision_penalty=0.0)
    raise NotImplementedError(f'reward {reward} not implemented')


def make_base_env(name: str, reward: str, collision_penalty: float, only_steering: bool):
    env = gym.make(name)
    if only_steering:
        env = FixSpeedControl(env, fixed_speed=2.0)
    env = get_reward_wrapper(reward, collision_penalty)(env)
    env = LidarOccupancyObservation(env, max_range=10.0, resolution=0.25)
    env = FilterObservationWrapper(env, obs_list=['lidar_occupancy', 'velocity'])
    env = FlattenAction(env)
    env = RescaleAction(env, a=-1, b=+1)
    env = NormalizeVelocityObservation(env)
    env = FrameSkip(env, frame_skip=4)
    return env
