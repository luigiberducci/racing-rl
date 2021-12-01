import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from racing_rl.envs.wrappers import FrameSkip
from skimage.color import rgb2gray


class VelocityWrapper(gym.ObservationWrapper):
    """ extend the observation space of racecar with velocity """

    def __init__(self, env):
        super(VelocityWrapper, self).__init__(env)
        self._w, self._h, c = self.observation_space.shape
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(80, 80, 1), dtype=np.uint8),
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


def twin_evaluations(model1, model2, env):
    """
    deterministic evaluation of 2 models when having the same observation as input.
    the action predicted by model1 is then used to interact with the environment.

    return a dictionary with the action for each model
    """
    done = False
    obs = env.reset()
    actions1, actions2 = [], []
    while not done:
        action1, _ = model1.predict(obs, deterministic=True)
        action2, _ = model2.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action1)
        actions1.append(action1)
        actions2.append(action2)
    return {'model1': np.array(actions1), 'model2': np.array(actions2)}


env = gym.make("CarRacing-v0")
env = VelocityWrapper(env)
env = FrameSkip(env, frame_skip=8)
check_env(env)
print("[info] env ok")

model = PPO("MultiInputPolicy", env, verbose=1)

model.learn(1000)
model.save("saved_model")

model2 = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=None)
model2.load("saved_model", print_system_info=True)

actions = twin_evaluations(model, model2, env)

import matplotlib.pyplot as plt

for i, name in enumerate(["steer", "gas", "brake"]):
    plt.subplot(3, 1, i + 1)
    plt.title(name)
    plt.plot(actions["model1"][:, i], label="after training")
    plt.plot(actions["model2"][:, i], label="after load")
    plt.legend()
plt.tight_layout()
plt.show()
