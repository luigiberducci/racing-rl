import argparse
import pathlib
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from racing_rl.envs.wrappers import FixResetWrapper
from racing_rl.training.agent_utils import CustomCNN, make_agent
from racing_rl.training.callbacks import VideoRecorderCallback
from racing_rl.training.env_utils import make_base_env

parser = argparse.ArgumentParser()
parser.add_argument("--track", type="str")
parser.add_argument("--n_steps", type="int")
parser.add_argument("--algo", choices=['sac', 'ppo'])
args = parser.parse_args()

# logs
task = f"SingleAgent{args.track.capitalize()}-v0"
timestamp = datetime.now.strftime("%m%d%Y_%H%M%S")
logdir = pathlib.Path(f"logs/{task}_{args.algo}_{timestamp}")

# make envs
train_env = make_base_env(task)
train_env = FixResetWrapper(mode="random")

eval_env = make_base_env(task)
eval_env = FixResetWrapper(mode="grid")

# callbacks
eval_callback = EvalCallback(eval_env, best_model_save_path=str(logdir / 'models'),
                             log_path=str(logdir / 'models'), eval_freq=1000,
                             deterministic=True, render=False)
video_recorder = VideoRecorderCallback(eval_env, render_freq=5000)
callbacks = [eval_callback, video_recorder]

# training
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)
model = make_agent(train_env, args.algo, logdir)

model.learn(args.n_steps, callback=callbacks)
