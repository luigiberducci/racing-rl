import argparse
import pathlib
from datetime import datetime

from gym.wrappers import Monitor, TimeLimit
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import CombinedExtractor

from racing_rl.envs.wrappers import FixResetWrapper, LapLimit, ElapsedTimeLimit
from racing_rl.training.agent_utils import CustomCNN, make_agent
from racing_rl.training.callbacks import VideoRecorderCallback
from racing_rl.training.env_utils import make_base_env

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=str, required=True)
parser.add_argument("--reward", type=str, required=True, choices=['min_steering', 'min_action', 'progress'])
parser.add_argument("--algo", choices=['sac', 'ppo', 'ddpg'], required=True)
parser.add_argument("--n_steps", type=int, default=1000000)
parser.add_argument("-only_steering", action='store_true')
args = parser.parse_args()

# logs
task = f"SingleAgent{args.track.capitalize()}-v0"
timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
logdir = pathlib.Path(f"logs/{task}_{args.reward}_{args.algo}_OnlySteering{args.only_steering}_{timestamp}")

# make envs
train_env = make_base_env(task, args.reward, only_steering=args.only_steering)
train_env = FixResetWrapper(train_env, mode="random")
train_env = TimeLimit(train_env, max_episode_steps=1000)

eval_task = f"SingleAgent{args.track.capitalize()}_Gui-v0"
eval_env = make_base_env(eval_task, 'progress', only_steering=args.only_steering)
eval_env = FixResetWrapper(eval_env, mode="grid")
eval_env = TimeLimit(eval_env, max_episode_steps=5000)
eval_env = Monitor(eval_env, logdir / 'videos')

# callbacks
eval_freq = 1000
eval_callback = EvalCallback(eval_env, best_model_save_path=str(logdir / 'models'),
                             log_path=str(logdir / 'evaluations'), eval_freq=eval_freq,
                             deterministic=True, render=True)
checkpoint_callback = CheckpointCallback(save_freq=eval_freq, save_path=str(logdir / 'models'))
#video_recorder = VideoRecorderCallback(eval_env, render_freq=10000)
callbacks = [eval_callback, checkpoint_callback]

# training
model = make_agent(train_env, args.algo, str(logdir))

model.learn(args.n_steps, callback=callbacks)
