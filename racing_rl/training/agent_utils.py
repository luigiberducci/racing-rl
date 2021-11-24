import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted. This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def make_agent(env, algo, logdir):
    if algo == 'sac':
        from stable_baselines3 import SAC
        # note: dealing wt img observation requires large amount of ram for replay buffer
        model = SAC("MultiInputPolicy", env, buffer_size=1000000,
                    verbose=1, tensorboard_log=logdir)
    elif algo == 'ppo':
        from stable_baselines3 import PPO
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir)
    elif algo == 'ddpg':
        from stable_baselines3 import DDPG
        n_actions = env.action_space.shape[-1]
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3 import HerReplayBuffer
        model = DDPG("MultiInputPolicy", env,
                     gamma=0.9,
                     learning_rate=0.00001,
                     learning_starts=1000,
                     gradient_steps=250,
                     batch_size=100,
                     verbose=1)
    else:
        raise NotImplementedError(algo)
    return model
