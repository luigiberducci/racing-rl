# experiment: ablation study on action space
# date: 30 Nov 2021
# aim:  evaluate the complexity of learning 1 control (steering) vs 2 controls (steering, speed)
#
# exp. setup: evaluate for only_steering=True (1 control) and only_steering=False (2 controls),
#             mdp with observation space (2d-occupancymap, velocity), reward 'min-action'
#             train PPO with MultiInput policy (CNN feature extractor for images -> concat -> MLP policy)
#             n_steps: 100K, n_seeds: 10

from argparse import Namespace

import numpy as np

from train import train

n_seeds = 10

params = {
    'track': "melbourne",
    'reward': "min_action",
    'algo': "ppo",
    'n_steps': 100000,
}

for i in range(n_seeds):
    for only_steering in [True, False]:
        current_params = params
        current_params['seed'] = np.random.randint(0, 1000000)
        current_params['only_steering'] = only_steering
        train(Namespace(**params))
