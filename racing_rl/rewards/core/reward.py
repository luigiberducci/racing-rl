from abc import ABC, abstractmethod
from typing import Any

import gym


class RewardFunction(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        pass


class RewardWrapper(gym.Wrapper):

    def __init__(self, env, reward_fn: RewardFunction):
        super().__init__(env)
        self._state = None
        self._reward = 0.0
        self._return = 0.0
        self._reward_fn = reward_fn

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self._state = state
        self._reward = 0.0
        self._return = 0.0
        return state

    def step(self, action: Any):
        next_state, _, done, info = self.env.step(action)
        reward = self._reward_fn(state=self._state, action=action, next_state=next_state, info=info)
        self._state = next_state
        self._reward = reward
        self._return += reward
        return next_state, reward, done, info

    def render(self, mode='human', **kwargs):
        return super(RewardWrapper, self).render(mode=mode, **kwargs)

