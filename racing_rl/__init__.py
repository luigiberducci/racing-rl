import pathlib
from racing_rl.envs.single_agent_env import SingleAgentRaceEnv
from racing_rl.envs.multi_agent_env import MultiAgentRaceEnv
from gym.envs.registration import register

track_dir = pathlib.Path(__file__).parent / ".." / "f1tenth_racetracks"

for track in track_dir.iterdir():
    track_name = str(track.stem).replace(" ", "")
    # single-agent
    register(
        id=f"SingleAgent{track_name}-v0",
        entry_point='racing_rl:SingleAgentRaceEnv',
        kwargs={'map_name': track_name, 'gui': False}
    )
    register(
        id=f"SingleAgent{track_name}_Gui-v0",
        entry_point='racing_rl:SingleAgentRaceEnv',
        kwargs={'map_name': track_name, 'gui': True}
    )
    # multi-agent
    register(
        id=f"MultiAgent{track_name}-v0",
        entry_point='racing_rl:MultiAgentRaceEnv',
        kwargs={'map_name': track_name, 'n_npcs': 1, 'gui': False}
    )
    register(
        id=f"MultiAgent{track_name}_Gui-v0",
        entry_point='racing_rl:MultiAgentRaceEnv',
        kwargs={'map_name': track_name, 'n_npcs': 1, 'gui': True}
    )


