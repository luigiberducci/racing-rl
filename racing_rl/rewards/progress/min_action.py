import numpy as np

from racing_rl.rewards.core.reward import RewardFunction


class MinActionReward(RewardFunction):

    def __init__(self, action_space, collision_penalty: float = 0.0):
        assert collision_penalty >= 0.0, f"penalty must be >=0 and will be subtracted to the reward ({collision_penalty}"
        self._collision_penalty = collision_penalty
        self._action_space = action_space
        super(MinActionReward, self).__init__()

    def _normalize_action(self, name, action_val):
        low, high = self._action_space[name].low, self._action_space[name].high
        norm_action = 2 * (action_val - low) / (high - low) - 1
        return norm_action

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert action is not None, "action not valid: None"
        action = [self._normalize_action(key, val) for key, val in action.items()]
        if info["collision"]:
            reward = - self._collision_penalty
        else:
            reward = 1 - (1 / len(action) * np.linalg.norm(action) ** 2)
        return reward
