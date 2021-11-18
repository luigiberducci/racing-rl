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


@dataclass
class Track:
    filepath: str
    ext: str
    occupancy_map: np.ndarray
    centerline: np.ndarray

    @staticmethod
    def from_track_name(track: str):
        try:
            track_dir = pathlib.Path(__file__).parent / ".." / "f1tenth_racetracks" / track
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

        """
        map_height = map_img.shape[0]
        map_width = map_img.shape[1]

        # convert map pixels to coordinates
        range_x = np.arange(map_width)
        range_y = np.arange(map_height)
        map_x, map_y = np.meshgrid(range_x, range_y)
        map_x = (map_x * map_resolution + origin_x).flatten()
        map_y = (map_y * map_resolution + origin_y).flatten()
        map_z = np.zeros(map_y.shape)
        map_coords = np.vstack((map_x, map_y, map_z))
        """


if __name__ == "__main__":
    track = Track.from_track_name("Catalunya")
    print("SUCCESS")
