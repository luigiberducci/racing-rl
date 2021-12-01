# experiment: ablation study on action space
# date: 30 Nov 2021
# aim:  evaluate the complexity of learning 1 control (steering) vs 2 controls (steering, speed)
#
# exp. setup: evaluate for only_steering=True (1 control) and only_steering=False (2 controls),
#             mdp with observation space (2d-occupancymap, velocity), reward 'min-action'
#             train PPO with MultiInput policy (CNN feature extractor for images -> concat -> MLP policy)
#             n_steps: 50K, n_seeds: 10
#
# how to run (2 separate instances of 10 seeds each):
#     $>python script/run_experiment_action_space.py --n_seeds 10 -only_steering
#     $>python script/run_experiment_action_space.py --n_seeds 10

import argparse
from argparse import Namespace

import numpy as np

from train import train

params = {
    'logdir': 'logs/experiments/actionspace',
    'track': "melbourne",
    'reward': "min_action",
    'collision_penalty': 10.0,
    'algo': "ppo",
    'n_steps': 100000,
}

parser = argparse.ArgumentParser()
parser.add_argument("--n_seeds", type=int, required=True)
parser.add_argument("-only_steering", action='store_true')
args = parser.parse_args()

for i in range(args.n_seeds):
    current_params = params
    current_params['seed'] = np.random.randint(0, 1000000)
    current_params['only_steering'] = args.only_steering
    train(Namespace(**params))
