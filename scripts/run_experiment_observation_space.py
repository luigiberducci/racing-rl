# experiment: experiments for various observation spaces
# date: 2 Dec 2021
# aim:  evaluate alternative solutions for remove velocity from the observations
#
# exp. setup: evaluate PPO under various observation-space formulations (including or excluding velocity)
#             mdp reward 'min-action', 2 controls (steer,speed)
#             train PPO with MultiInput policy or MlpPolicy (depending on the observations)
#             n_steps: 250K, n_seeds: 1
#
# how to run (3 separate instances and 1 seed each):
#     $>python script/run_experiment_observation_space.py --n_seeds 1 -include_velocity
#     $>python script/run_experiment_observation_space.py --n_seeds 1 --frame_aggr max
#     $>python script/run_experiment_observation_space.py --n_seeds 1 --frame_aggr stack

import argparse
from argparse import Namespace

import numpy as np

from train import train

params = {
    'logdir': 'logs/experiments/observationspace',
    'track': 'melbourne',
    'reward': 'min_action',
    'collision_penalty': 10.0,
    'algo': 'ppo',
    'only_steering': False,
    'n_steps': 250000,
}

parser = argparse.ArgumentParser()
parser.add_argument("--n_seeds", type=int, required=True)
parser.add_argument("-include_velocity", action='store_true', help="include velocity in the observation")
parser.add_argument("--frame_aggr", choices=['max', 'stack'], default=None, help="used if velocity not observed")
args = parser.parse_args()

for _ in range(args.n_seeds):
    current_params = params
    current_params['seed'] = np.random.randint(0, 1000000)
    current_params['include_velocity'] = args.include_velocity
    current_params['frame_aggr'] = args.frame_aggr
    train(Namespace(**params))
