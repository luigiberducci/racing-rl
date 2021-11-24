import argparse
import pathlib
from datetime import datetime

from gym.wrappers import Monitor, TimeLimit
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import CombinedExtractor

from racing_rl.envs.wrappers import FixResetWrapper, LapLimit, ElapsedTimeLimit
from racing_rl.training.agent_utils import CustomCNN, make_agent
from racing_rl.training.callbacks import VideoRecorderCallback
from racing_rl.training.env_utils import make_base_env

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=str, required=True)
parser.add_argument("--algo", choices=['sac', 'ppo', 'ddpg'], required=True)
parser.add_argument("--n_steps", type=int, default=5000000)
args = parser.parse_args()

# logs
task = f"SingleAgent{args.track.capitalize()}-v0"
timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
logdir = pathlib.Path(f"logs/{task}_{args.algo}_{timestamp}")

# make envs
train_env = make_base_env(task)
train_env = FixResetWrapper(train_env, mode="random")
train_env = TimeLimit(train_env, max_episode_steps=2000)

eval_task = f"SingleAgent{args.track.capitalize()}_Gui-v0"
eval_env = make_base_env(eval_task)
eval_env = FixResetWrapper(eval_env, mode="grid")
eval_env = TimeLimit(eval_env, max_episode_steps=12000)
eval_env = LapLimit(eval_env, max_episode_laps=1)
eval_env = Monitor(eval_env, logdir / 'videos')

# callbacks
eval_callback = EvalCallback(eval_env, best_model_save_path=str(logdir / 'models'),
                             log_path=str(logdir / 'models'), eval_freq=1000,
                             deterministic=True, render=True)
#video_recorder = VideoRecorderCallback(eval_env, render_freq=10000)
callbacks = [eval_callback]

# training
policy_kwargs = dict(
    features_extractor_class=CombinedExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)
model = make_agent(train_env, args.algo, str(logdir))

model.learn(args.n_steps, callback=callbacks)
