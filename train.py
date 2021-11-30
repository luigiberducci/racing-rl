import pathlib
from datetime import datetime

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
    plt.subplot(2, 1, 1)
    plt.title("steering")
    plt.plot(actions[:, 0])
    plt.ylim(-1, +1)
    plt.subplot(2, 1, 2)
    plt.title("velocity")
    plt.plot(actions[:, 1])
    plt.ylim(-1, +1)
    plt.savefig(str(logdir / "action_distribution.pdf"))


def train(args):
    # logs
    task = f"SingleAgent{args.track.capitalize()}-v0"
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    logdir = pathlib.Path(
        f"logs/{args.track}_{args.reward}_{args.algo}_OnlySteering{args.only_steering}_Seed{args.seed}_{timestamp}")

    # set seed for reproducibility
    utils.seeding(args.seed)

    # make envs
    train_env = make_base_env(task, args.reward, only_steering=args.only_steering)
    train_env = FixResetWrapper(train_env, mode="random")
    train_env = TimeLimit(train_env, max_episode_steps=1000)

    eval_task = f"SingleAgent{args.track.capitalize()}-v0"
    eval_env = make_base_env(eval_task, 'only_progress', only_steering=args.only_steering)
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
    parser.add_argument("--track", type=str, required=True)
    parser.add_argument("--reward", type=str, required=True, choices=['min_steering', 'min_action', 'progress'])
    parser.add_argument("--algo", choices=['sac', 'ppo', 'ddpg'], required=True)
    parser.add_argument("--n_steps", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-only_steering", action='store_true')
    args = parser.parse_args()
    train(args)
