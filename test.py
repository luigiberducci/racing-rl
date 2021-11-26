import argparse
import pathlib

import numpy as np
from gym.wrappers import TimeLimit

from racing_rl.envs.wrappers import FixResetWrapper, LapLimit
from racing_rl.training.agent_utils import make_agent
from racing_rl.training.env_utils import make_base_env

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=str, required=True)
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--n_episodes", type=int, default=5)
args = parser.parse_args()


def find_algo(path: pathlib.Path):
    for algo in ['sac', 'ppo', 'ddpg']:
        if algo in str(path):
            return algo
    raise NotImplementedError(f'not able to extract model from {str(path)}')


def find_if_onlysteering(path: pathlib.Path):
    return 'OnlySteeringTrue' in str(path)


algo = find_algo(args.checkpoint)
onlysteering = find_if_onlysteering(args.checkpoint)

#eval_task = f"SingleAgent{args.track.capitalize()}_Gui-v0"
#eval_env = make_base_env(eval_task, reward='progress', only_steering=onlysteering)
#eval_env = FixResetWrapper(eval_env, mode='grid')
#eval_env = LapLimit(eval_env, max_episode_laps=1)

task = f"SingleAgent{args.track.capitalize()}-v0"
eval_env = make_base_env(task, 'only_progress', only_steering=onlysteering)
eval_env = FixResetWrapper(eval_env, mode="grid")
eval_env = TimeLimit(eval_env, max_episode_steps=1000)

print(algo)
for t in range(5):
    print(t)
    model = make_agent(eval_env, algo, logdir=None)
    model.load(str(args.checkpoint))

    progresses = []
    for e in range(args.n_episodes):
        done = False
        obs = eval_env.reset()
        ret, step, progress_t0 = 0.0, 0, -1.0
        while not done:
            assert obs['lidar_occupancy'].shape[0] == 1
            action, _ = model.predict(obs, deterministic=True)
            action = np.array([action[0], -1])
            obs, reward, done, info = eval_env.step(action)
            if progress_t0 < 0:
                progress_t0 = info['progress']
            ret += reward
            step += 1
            eval_env.render()
        progress = info['progress'] - progress_t0
        print(f"[Info] Episode {e + 1}, steps: {step}, progress: {progress:.3f}")
        progresses.append(progress)

    print(f"[Result] avg progress: {sum(progresses) / len(progresses):.3f}")
