from __future__ import annotations
from typing import Protocol, Tuple
import numpy as np
from .interfaces import Snapshot

class Featurizer(Protocol):
    """Encodes a Snapshot into an observation array."""
    def shape(self, grid_h: int, grid_w: int) -> Tuple[int, ...]: ...
    def encode(self, snap: Snapshot) -> np.ndarray: ...
