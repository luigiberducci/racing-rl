import gym
from gym.wrappers import RescaleAction, TimeLimit
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from racing_rl.envs.wrappers import LidarOccupancyObservation, FilterObservationWrapper, FlattenAction, FixSpeedControl, \
    FrameSkip, NormalizeVelocityObservation, ConstrainedSpeedControl
from racing_rl.rewards.min_action import MinSteeringReward, MinActionReward
from racing_rl.rewards.progress_based import ProgressReward


def get_reward_wrapper(reward: str):
    if reward == 'min_steering':
        return MinSteeringReward
    elif reward == 'min_action':
        return MinActionReward
    elif reward == 'progress':
        return lambda env: ProgressReward(env, env.track, with_penalty=True)
    elif reward == 'only_progress':
        return lambda env: ProgressReward(env, env.track)
    raise NotImplementedError(f'reward {reward} not implemented')


def make_base_env(name: str, reward: str, only_steering: bool = False):
    env = gym.make(name)
    if only_steering:
        env = FixSpeedControl(env, fixed_speed=2.0)
    env = get_reward_wrapper(reward)(env)
    env = LidarOccupancyObservation(env, max_range=10.0, resolution=0.25)
    env = FilterObservationWrapper(env, obs_list=['lidar_occupancy', 'velocity'])
    env = FlattenAction(env)
    env = RescaleAction(env, a=-1, b=+1)
    env = NormalizeVelocityObservation(env)
    env = FrameSkip(env, frame_skip=4)
    return env