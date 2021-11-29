import pathlib
import time
from collections import OrderedDict

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from racing_rl.envs.wrappers import FrameSkip
from skimage.color import rgb2gray


class VelocityWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(VelocityWrapper, self).__init__(env)
        self._w, self._h, c = self.observation_space.shape
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(80, 80, 1)),
            'velocity': gym.spaces.Box(low=np.NINF, high=np.PINF, shape=(2,))
        })

    def observation(self, observation):
        image = np.reshape((rgb2gray(observation) * 255).astype(np.uint8), (self._w, self._h, 1))[:80, :80, :]
        image = np.where(image <= 150, 0, 255)
        dict_observation = {
            'image': image,
            'velocity': np.array(self.env.car.hull.linearVelocity)
        }
        return dict_observation


class SkipFirstKFrames(gym.Wrapper):
    def __init__(self, env, k_frames):
        self._k_frames = k_frames
        super(SkipFirstKFrames, self).__init__(env)

    def reset(self, **kwargs):
        obs = super(SkipFirstKFrames, self).reset(**kwargs)
        for k in range(self._k_frames):
            obs, _, _, _ = self.step(np.array([0.0, 0.0, 0.0]))
        return obs


class ReduceActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ReduceActionSpaceWrapper, self).__init__(env)
        self.action_space = gym.spaces.Box(low=-1, high=+1, shape=(1,))

    def action(self, action):
        acceleration = 0.1 if self.env.car.hull.linearVelocity[0] < 1.0 else 0.0
        return np.concatenate([action, np.array([acceleration, 0.0])])


class DiscreteActionSpaceWrapper(gym.ActionWrapper):
    ACTIONS = {0: np.array([-1.0, 0.0, 0.0]),  # turn left
               1: np.array([+1.0, 0.0, 0.0]),  # turn right
               2: np.array([0.0, 0.0, 0.8]),  # brake
               3: np.array([0.0, 1.0, 0.8]),  # accelerate
               4: np.array([0.0, 0.0, 0.0])}  # do nothing

    def __init__(self, env):
        super(DiscreteActionSpaceWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(5)

    def action(self, action):
        return self.ACTIONS[action]

def mkenv():
    env = gym.make("CarRacing-v0")
    env = SkipFirstKFrames(env, k_frames=25)
    env = VelocityWrapper(env)
    env = DiscreteActionSpaceWrapper(env)
    env = FrameSkip(env, frame_skip=8)
    check_env(env)
    return env

env = DummyVecEnv([lambda: mkenv()])
print("[info] env ok")

logdir = pathlib.Path(f"check_error/{time.time()}")
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=str(logdir))

freq = 10000
model.learn(50000, callback=[EvalCallback(env, render=True, deterministic=True, eval_freq=freq, n_eval_episodes=1),
                             CheckpointCallback(save_freq=freq, save_path=str(logdir / "models"))])
mean, std = evaluate_policy(model, env, 5, deterministic=True)
print(f"after training: mean={mean} +/- {std}")

model.save(str(logdir / "models" / "final_model"))

model2 = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=None)
model2.load(str(logdir / "models" / "final_model"))
mean, std = evaluate_policy(model2, env, 5, deterministic=True)
print(f"after loading: mean={mean} +/- {std}")
