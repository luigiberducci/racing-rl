import pathlib

import gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from racing_rl.envs.wrappers import FixResetWrapper
from racing_rl.training.env_utils import make_base_env

nagents = 2
name = 'MultiAgentHoorsaal-v0'
env, env_params = make_base_env(name, reward="hrs_conservative", collision_penalty=0.0,
                                only_steering=False, include_velocity=True, frame_aggregation='stack',
                                curv_control=False)

name = 'MultiAgentHoorsaal_Gui-v0'
evalenv, _ = make_base_env(name, reward="only_progress", collision_penalty=0.0,
                           only_steering=False, include_velocity=True, frame_aggregation='stack',
                           curv_control=False)
env = FixResetWrapper(env, 'grid')

logdir = pathlib.Path("logs/f110-multiagent")
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=str(logdir / "models"),
                                         name_prefix='model')
model = SAC("MultiInputPolicy", evalenv, verbose=1, tensorboard_log=str(logdir))
#model.learn(2000000, eval_env=evalenv, eval_freq=10000,
#            n_eval_episodes=5, callback=checkpoint_callback)
model.save(str(logdir / "models/final" ))

model = SAC.load("/home/luigi/Development/f1tenth/racing-rl/logs/f110-multiagent/models/model_1000000_steps.zip")

for i in range(100):
    done = False

    obs = evalenv.reset()
    rrt = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = evalenv.step(action)
        rrt += reward
    print(f"episode {i}: retun: {rrt:.2f}")
