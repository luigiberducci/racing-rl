import pathlib
from datetime import datetime

import yaml
import numpy as np

from gym.wrappers import Monitor, TimeLimit
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from racing_rl.envs.wrappers import FixResetWrapper
from racing_rl.training.agent_utils import make_agent, evaluate_action_distribution
from racing_rl.training.env_utils import make_base_env
from racing_rl.training import utils


def save_action_figure(actions: np.ndarray, logdir: pathlib.Path):
    import matplotlib.pyplot as plt
    plt.clf()
    action_names = [action_name for _, action_name in zip(range(actions.shape[1]), ["steering", "velocity"])]
    for i, action_name in enumerate(action_names):
        plt.subplot(len(action_names), 1, i+1)
        plt.title(action_name)
        plt.plot(actions[:, i])
        plt.ylim(-1, +1)
    plt.savefig(str(logdir / "action_distribution.pdf"))


def save_params(logdir, args):
    assert type(args) == dict
    logdir.mkdir(parents=True, exist_ok=True)
    with open(str(logdir / "args.yaml"), "w") as outfile:
        yaml.dump(dict(args), outfile)


def train(args):
    # logs
    task = f"SingleAgent{args.track.capitalize()}-v0"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = pathlib.Path(
        f"{args.logdir}/{args.track}_{args.algo}_OnlySteering{args.only_steering}_{args.reward}_CollPenalty{args.collision_penalty}_{timestamp}")
    save_params(logdir, vars(args))

    # set seed for reproducibility
    utils.seeding(args.seed)

    # make envs
    train_env = make_base_env(task, args.reward, collision_penalty=args.collision_penalty,
                              only_steering=args.only_steering)
    train_env = FixResetWrapper(train_env, mode="random")
    train_env = TimeLimit(train_env, max_episode_steps=1000)

    eval_task = f"SingleAgent{args.track.capitalize()}-v0"
    eval_env = make_base_env(eval_task, 'only_progress', collision_penalty=0.0, only_steering=args.only_steering)
    eval_env = FixResetWrapper(eval_env, mode="grid")
    eval_env = TimeLimit(eval_env, max_episode_steps=5000)
    eval_env = Monitor(eval_env, logdir / 'videos')

    # callbacks
    eval_freq = 5000
    eval_callback = EvalCallback(eval_env, best_model_save_path=str(logdir / 'models'), n_eval_episodes=3,
                                 log_path=str(logdir / 'evaluations'), eval_freq=eval_freq,
                                 deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=eval_freq, save_path=str(logdir / 'models'))
    callbacks = [eval_callback, checkpoint_callback]

    # training
    model = make_agent(train_env, args.algo, str(logdir))
    model.learn(args.n_steps, callback=callbacks)

    # evaluate trained policy
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"[after training] mean_reward={mean_reward:.2f} +/- {std_reward}")
    # save (apparently not working, todo discuss wt Axel)
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
    parser.add_argument("--n_steps", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-only_steering", action='store_true')
    args = parser.parse_args()
    train(args)
