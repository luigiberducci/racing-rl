import time

from gym.wrappers import TimeLimit
import matplotlib.pyplot as plt

from racing_rl.baseline.PurePursuitPlanner import PurePursuitPlanner
from racing_rl.envs.wrappers import FixResetWrapper, LapLimit, ElapsedTimeLimit, LidarOccupancyObservation, \
    FixSpeedControl, TerminateOnlyOnTimeLimit
from racing_rl.rewards.progress_based import ProgressReward
from racing_rl.training.env_utils import make_base_env
import racing_rl
import gym

env = gym.make('SingleAgentMelbourne_Gui-v0')
env = FixSpeedControl(env, 1.0)
env = ProgressReward(env, env.track)
env = LidarOccupancyObservation(env, max_range=10.0, resolution=0.25)
env = FixResetWrapper(env, mode='random')
env = TerminateOnlyOnTimeLimit(env, max_episode_steps=1000)


def render_callback(env_renderer):
    # custom extra drawing function
    e = env_renderer

    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 800
    e.right = right + 800
    e.top = top + 800
    e.bottom = bottom - 800

    planner.render_waypoints(env_renderer)


env.add_render_callback(render_callback)

planner = PurePursuitPlanner(env.track, wb=0.17145 + 0.15875, fixed_speed=5.0)

for i in range(5):
    t0 = time.time()
    obs = env.reset()
    done = False
    t = 0
    while not done:
        t += 1
        speed, steer = planner.plan(obs['pose'][0], obs['pose'][1], obs['pose'][2], lookahead_distance=1.5, vgain=1.0)
        obs, reward, done, info = env.step({'steering': steer, 'velocity': speed})
        # print(reward)
        #env.render()
        #if t % 10 == 0:
        #    plt.clf()
        #    plt.imshow(obs['lidar_occupancy'])
        #    plt.pause(0.01)
    print(f"DONE, sim time: {info['lap_time']}, real time: {time.time() - t0}")
