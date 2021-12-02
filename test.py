import argparse
import pathlib

import numpy as np
from gym.wrappers import TimeLimit
from matplotlib import pyplot as plt

from racing_rl.envs.wrappers import FixResetWrapper, LapLimit
from racing_rl.training.agent_utils import make_agent
from racing_rl.training.env_utils import make_base_env

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=str, required=True)
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--n_episodes", type=int, default=5)
parser.add_argument("-render_obs", action='store_true', help='enable rendering of the occupancy-map from lidar')
args = parser.parse_args()


def find_algo(path: pathlib.Path):
    for algo in ['sac', 'ppo', 'ddpg']:
        if algo in str(path):
            return algo
    raise NotImplementedError(f'not able to extract model from {str(path)}')


def find_if_onlysteering(path: pathlib.Path):
    return 'OnlySteeringTrue' in str(path)


def find_if_include_velocity(path: pathlib.Path):
    include_velocity = 'ObsVelocityTrue' in str(path)
    frame_aggr = None
    if 'AggrFrameMax' in str(path):
        frame_aggr = "max"
    elif 'AggrFrameStack' in str(path):
        frame_aggr = "stack"
    assert include_velocity or frame_aggr, "assertion: not(include_velocity) -> (frame_aggregator!=None)"
    return include_velocity, frame_aggr


algo = find_algo(args.checkpoint)
onlysteering = find_if_onlysteering(args.checkpoint)
include_velocity, frame_aggr = find_if_include_velocity(args.checkpoint)

task = f"SingleAgent{args.track.capitalize()}-v0"
eval_env, _ = make_base_env(task, 'only_progress', collision_penalty=0.0, only_steering=onlysteering,
                            include_velocity=include_velocity, frame_aggregation=frame_aggr)
eval_env = FixResetWrapper(eval_env, mode="grid")
eval_env = LapLimit(eval_env, max_episode_laps=1)

print(algo)
for t in range(args.n_episodes):
    print(f"episode {t + 1}")
    model, _ = make_agent(eval_env, algo, logdir=None)
    model = model.load(str(args.checkpoint))

    progresses = []
    for e in range(args.n_episodes):
        done = False
        obs = eval_env.reset()
        ret, step, progress_t0 = 0.0, 0, -1.0
        steerings, speeds, velocities = [], [], []
        while not done:
            assert obs['lidar_occupancy'].shape[0] == 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            if progress_t0 < 0:
                progress_t0 = info['progress']
            ret += reward
            step += 1
            # collect action statistics
            steerings.append(info['action']['steering'])
            speeds.append(info['action']['velocity'])
            velocities.append(info['velocity'])
            # rendering
            eval_env.render()
            if args.render_obs and step % 25 == 0:
                plt.clf()
                plt.imshow(obs['lidar_occupancy'][0])
                plt.pause(0.001)
        # print results
        progress = info['progress'] - progress_t0   # note: it is faulty when crossing starting line
        progresses.append(progress)
        print(f"[Info] Episode {e + 1}, steps: {step}, progress: {progress:.3f}")
        # plot actions
        for name, array in zip(['steering_cmd', 'speed_cmd', 'velocity'], [steerings, speeds, velocities]):
            array = np.array(array)
            plt.plot(array, label=name)
        plt.legend()
        plt.show()
    print(f"[Result] avg progress: {sum(progresses) / len(progresses):.3f}")
