import gym
import f110_gym
import numpy as np

nagents = 2
env = gym.make('f110_gym:f110-v0', num_agents=nagents)
done = False

obs = env.reset(np.array([[0.0, 0.0] for _ in range(nagents)]))

while not done:
    obs, reward, done, info = env.step([0.0, np.random.random()] for _ in range(nagents))
    env.render()