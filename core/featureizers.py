from __future__ import annotations
import numpy as np
from .feature_iface import Featurizer
from .interfaces import Snapshot

class CH3Featurizer(Featurizer):
    def __init__(self, include_walls: bool = True, include_dir: bool = True, flat: bool = False):
        self.include_walls = include_walls
        self.include_dir = include_dir
        self.flat = flat

    def shape(self, grid_h: int, grid_w: int):
        C = 3 + (1 if self.include_walls else 0) + (2 if self.include_dir else 0)
        return (grid_h*grid_w*C,) if self.flat else (grid_h, grid_w, C)

    def encode(self, snap: Snapshot) -> np.ndarray:
        # can reuse your env’s _encode_obs logic here later
        raise NotImplementedError
