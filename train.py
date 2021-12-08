import pathlib
import time
from datetime import datetime

import gym
import yaml
import numpy as np

from gym.wrappers import Monitor, TimeLimit
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from racing_rl.envs.wrappers import FixResetWrapper, LapLimit
from racing_rl.training.agent_utils import make_agent, evaluate_action_distribution
from racing_rl.training.env_utils import make_base_env
from racing_rl.training import utils


def save_action_figure(actions: np.ndarray, logdir: pathlib.Path):
    import matplotlib.pyplot as plt
    plt.clf()
    action_names = [action_name for _, action_name in zip(range(actions.shape[1]), ["steering", "velocity"])]
    for i, action_name in enumerate(action_names):
        plt.subplot(len(action_names), 1, i + 1)
        plt.title(action_name)
        plt.plot(actions[:, i])
        plt.ylim(-1, +1)
    plt.savefig(str(logdir / "action_distribution.pdf"))


def save_params(logdir, args, filename):
    assert type(args) == dict
    logdir.mkdir(parents=True, exist_ok=True)
    with open(str(logdir / f"{filename}.yaml"), "w") as outfile:
        yaml.dump(dict(args), outfile)
        print(f"[Info] Saved parameters in {outfile.name}")


def train(args):
    # logs
    task = f"SingleAgent{args.track.capitalize()}-v0"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = pathlib.Path(
        f"{args.logdir}/{args.track}_{args.algo}_ObsVelocity{args.include_velocity}_AggrFrames{str(args.frame_aggr).capitalize()}_OnlySteering{args.only_steering}_{args.reward}_CollPenalty{args.collision_penalty}_{timestamp}")
    save_params(logdir, vars(args), filename="args")

    # set seed for reproducibility
    utils.seeding(args.seed)

    # make envs
    train_env, trainenv_params = make_base_env(task, args.reward, collision_penalty=args.collision_penalty,
                                               only_steering=args.only_steering, include_velocity=args.include_velocity,
                                               frame_aggregation=args.frame_aggr)
    train_env = FixResetWrapper(train_env, mode="random")
    train_env = TimeLimit(train_env, max_episode_steps=1000)
    save_params(logdir, trainenv_params, filename="training_env")

    eval_task = f"SingleAgent{args.track.capitalize()}_Gui-v0"
    eval_env, _ = make_base_env(eval_task, 'only_progress', collision_penalty=0.0,
                                only_steering=args.only_steering, include_velocity=args.include_velocity,
                                frame_aggregation=args.frame_aggr)
    eval_env = FixResetWrapper(eval_env, mode="grid")
    eval_env = TimeLimit(eval_env, max_episode_steps=5000)
    eval_env = LapLimit(eval_env, max_episode_laps=1)
    eval_env = Monitor(eval_env, logdir / 'videos')

    # callbacks
    eval_freq = 5000
    eval_callback = EvalCallback(eval_env, best_model_save_path=str(logdir / 'models'), n_eval_episodes=5,
                                 log_path=str(logdir / 'evaluations'), eval_freq=eval_freq,
                                 deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=eval_freq, save_path=str(logdir / 'models'))
    callbacks = [eval_callback, checkpoint_callback]

    # training
    model, model_params = make_agent(train_env, args.algo, str(logdir))
    save_params(logdir, model_params, "training_agent")
    model.learn(args.n_steps, callback=callbacks)

    # evaluate trained policy and store final version
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"[after training] mean_reward={mean_reward:.2f} +/- {std_reward}")
    model.save(str(logdir / 'models' / f'final_model_{int(mean_reward)}'))

    # evaluate action distribution
    actions = evaluate_action_distribution(model, eval_env, deterministic=True)
    save_action_figure(actions, logdir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--track", type=str, required=True)
    parser.add_argument("--reward", type=str, required=True, choices=['min_action', 'progress'])
    parser.add_argument("--algo", choices=['sac', 'ppo', 'ddpg'], required=True)
    parser.add_argument("--collision_penalty", type=float, default=10.0)
    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame_aggr", choices=['max', 'stack'], default=None, help="used if velocity not observed")
    parser.add_argument("-only_steering", action='store_true', help="reduce control to only the steering command")
    parser.add_argument("-include_velocity", action='store_true', help="include velocity in the observation")
    args = parser.parse_args()

    t0 = time.time()
    train(args)
    print(f"\n[info] done in {int(time.time() - t0)} seconds")
