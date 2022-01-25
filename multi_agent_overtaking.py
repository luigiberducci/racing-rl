import gym
from stable_baselines3 import PPO

from racing_rl.training.env_utils import make_base_env

nagents = 2
name = 'MultiAgentMelbourne-v0'
env, env_params = make_base_env(name, reward="hrs_conservative", collision_penalty=0.0,
                                only_steering=False, include_velocity=True, frame_aggregation='stack')

name = 'MultiAgentMelbourne_Gui-v0'
evalenv, _ = make_base_env(name, reward="only_progress", collision_penalty=0.0,
                           only_steering=False, include_velocity=True, frame_aggregation='stack')

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(100000, eval_env=evalenv, eval_freq=10000)

for i in range(100):
    print(i)
    done = False

    obs = env.reset()

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(reward)
