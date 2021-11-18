import gym
import racing_rl

env = gym.make("SingleAgentCatalunya-v0")

for i in range(5):
    obs = env.reset(mode='grid')
    done = False
    t = 0
    while t<1000 and not done:
        t += 1
        obs, reward, done, info = env.step(env.action_space.sample())
        if obs['collision']:
            print("COLLISION")
        env.render()