import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import yaml
from PIL import Image
from yamldataclassconfig.config import YamlDataClassConfig


@dataclass
class TrackSpec(YamlDataClassConfig):
    image: str
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float


def find_track_dir(track_name):
    # we assume there are no blank space in the track name. however, to take into account eventual blank spaces in
    # the map dirpath, we loop over all possible maps and check if there is a matching with the current track
    base_dir = pathlib.Path(__file__).parent / ".." / ".." / "f1tenth_racetracks"
    for dir in base_dir.iterdir():
        if track_name == str(dir.stem).replace(" ", ""):
            return dir
    raise FileNotFoundError(f'no mapdir matching {track_name} in {base_dir}')


@dataclass
class Track:
    filepath: str
    ext: str
    occupancy_map: np.ndarray
    centerline: np.ndarray

    @staticmethod
    def from_track_name(track: str):
        try:
            track_dir = find_track_dir(track)
            # load track spec
            with open(track_dir / f"{track}_map.yaml", 'r') as yaml_stream:
                map_metadata = yaml.safe_load(yaml_stream)
                track_spec = TrackSpec(**map_metadata)
            # load occupancy grid
            map_filename = pathlib.Path(track_spec.image)
            occupancy_map = np.array(Image.open(track_dir / str(map_filename)).
                                     transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
            # load centerline
            centerline = np.loadtxt(track_dir / f"{track}_centerline.csv", delimiter=',')
        except Exception as ex:
            print(f"map {track} not found\n{ex}")
            exit(-1)
        return Track(filepath=str((track_dir / map_filename.stem).absolute()), ext=map_filename.suffix,
                     occupancy_map=occupancy_map, centerline=centerline)


if __name__ == "__main__":
    track = Track.from_track_name("MexicoCity")
    print("[Result] map loaded successfully")
