import gym
import numpy as np


class ActionCurvature(gym.ActionWrapper):
    def __init__(self, env, wheelbase):
        super(ActionCurvature, self).__init__(env)
        self._wb = wheelbase
        min_curvature, max_curvature = -1.5, 1.5  # assuming wb ~ 33cm, max steering 0.31 rad can achieve curvature 1.31
        min_velocity, max_velocity = 0.5, 5.0
        self.action_space = gym.spaces.Dict({
            "curvature": gym.spaces.Box(low=min_curvature, high=max_curvature, shape=()),
            "velocity": gym.spaces.Box(low=min_velocity, high=max_velocity, shape=())
        })

    def action(self, action):
        """ input is a predicted curvature, the original action is a steering and velocity command """
        original_action = {"steering": np.arctan(self._wb * action["curvature"]),
                           "velocity": action["velocity"]}
        return original_action