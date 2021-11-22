import time

from gym.wrappers import TimeLimit

from racing_rl.envs.wrappers import FixResetWrapper, LapLimit, ElapsedTimeLimit
from racing_rl.training.env_utils import make_base_env

env = make_base_env('SingleAgentMelbourne-v0')
env = FixResetWrapper(env, mode='random')
env = ElapsedTimeLimit(env, max_episode_duration=10.0)

for i in range(5):
    t0 = time.time()
    obs = env.reset()
    done = False
    t = 0
    while not done:
        t += 1
        obs, reward, done, info = env.step([0.0, -1.0])
        #print(reward)
        if info['collision']:
            print("COLLISION")
        env.render()
    print(f"DONE, sim time: {info['lap_time']}, real time: {time.time() - t0}")