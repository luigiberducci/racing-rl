# racing-rl
reinforcement learning for f1tenth racing

# how to start
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
   pip install --user -r requirements.txt
   ```