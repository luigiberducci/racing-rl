# Racing-RL
reinforcement learning for f1tenth racing

# How to start
The implementation has been tested with `Python 3.9` under `Ubuntu 20.04`.

Installation:
1. Clone this repo.
2. Then initialize submodules `f1tenth_gym` and `f1tenth_racetracks`:
    ```
    git submodule init
    git submodule update
    ```
3. Install `f1tenth_gym`:
   ```
   cd f1tenth_gym
   pip install --user -e gym
   ```
4. Install requirements:
   ```
   pip install -r requirements.txt
   ```
5. Run a sample training:
    ```
    python train.py --track melbourne --algo ppo --reward min_action -include_velocity
    ```
   This will run a default training configuration, in `20K` steps (approx. 7 min)
   
# Current status
- Observation space
  - 2d-map from lidar scan + velocity
  - 2d-map from lidar scan + max frame-aggregation (like atari)
  - 2d-map from lidar scan + stack frame-aggregation (on channel dim)
- Action space (def. in `racing_rl/envs/single_agent_env.py:action_space`)
  - only steering: steering `+/-0.41 rad`, fixed speed = `2 m/s`
  - both controls: steering `+/-0.41 rad`, speed `[0,10] m/s`
- Reward definitions
  - progress (`racing_rl/rewards/progress_based.py`): 
    - the reward is proportional to its progress w.r.t. the centerline, optionally using a penalty for collision   
  - min_action (`racing_rl/rewards/min_action.py`): 
    - the reward is inversely proportional to the actions' deviation from the mid-value of the action domain (steering=0.0 rad, speed=5.0 m/s)

# TODO
- [ ] add render-mode `rgb_array` to store video during the training process
- [ ] `track.get_progress` does not correctly manage the crossing of the starting-line (from `0.99` to `1.99`)
- [ ] find stable problem configuration w.r.t. the following questions:
  - [ ] what is the minimal observation space? (ideally only lidar-based)
  - [ ] what is the less-restrictive action space? (ideally constrained only by action ranges)
  - [ ] what is a simple reward that enable good training?
- [ ] refactor the code structure, e.g., `make_base_env` is getting messy with a lot of wrappers
- [ ] tune base algorithms