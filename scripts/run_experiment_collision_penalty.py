# experiment: experiments for various collision_penalty in reward shaping
# date: 1 Dec 2021
# aim:  evaluate the impact of the collision penalty in the reward shaping
#
# exp. setup: evaluate PPO under various reward formulations (0.0, 1.0, 5.0, 10.0, 15.0, 20.0)
#             mdp with observation space (2d-occupancymap, velocity), reward 'min-action', 2 controls (steer,speed)
#             train PPO with MultiInput policy (CNN feature extractor for images -> concat -> MLP policy)
#             n_steps: 50K, n_seeds: 10
#
# how to run (2 separate instances of 3 penalties and 5 seeds each):
#     $>python script/run_experiment_collision_penalty.py --n_seeds 5 --collision_penalty 0.0 1.0 5.0
#     $>python script/run_experiment_collision_penalty.py --n_seeds 5 --collision_penalty 10.0 15.0 20.0

import argparse
from argparse import Namespace

import numpy as np

from train import train

params = {
    'logdir': 'logs/experiments/collisionpenalty',
    'track': 'melbourne',
    'reward': 'min_action',
    'algo': 'ppo',
    'only_steering': False,
    'n_steps': 50000,
}

parser = argparse.ArgumentParser()
parser.add_argument("--n_seeds", type=int, required=True)
parser.add_argument("--collision_penalty", type=float, nargs='+', required=True)
args = parser.parse_args()

for collision_penalty in args.collision_penalty:
    for _ in range(args.n_seeds):
        current_params = params
        current_params['seed'] = np.random.randint(0, 1000000)
        current_params['collision_penalty'] = collision_penalty
        train(Namespace(**params))
