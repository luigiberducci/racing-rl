import pathlib
from racing_rl.envs.single_agent_env import SingleAgentRaceEnv
from gym.envs.registration import register

track_dir = pathlib.Path(__file__).parent / ".." / "f1tenth_racetracks"

for track in track_dir.iterdir():
    track_name = str(track.stem).replace(" ", "")
    register(
        id=f"SingleAgent{track_name}-v0",
        entry_point='racing_rl:SingleAgentRaceEnv',
        kwargs={'map_name': track_name}
    )

