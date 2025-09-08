# interfaces/__init__.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Protocol
import numpy as np

# from core/interfaces.py
@dataclass(frozen=True)
class Snapshot:
    snake: Tuple[Tuple[int,int], ...]   # head first
    food: Tuple[int,int]
    dir: Tuple[int,int]
    score: int
    steps_since_food: int
    step_count: int
    terminated: bool
    reason: str | None
    grid_w: int
    grid_h: int

@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]  # should include {"snapshot": Snapshot, "score": int, "reason": str|None}

class Env(Protocol):
    def reset(self, *, seed: int | None = None) -> np.ndarray: ...
    def step(self, action: int) -> StepResult: ...
    def action_space_n(self) -> int: ...
    def observation_shape(self) -> Tuple[int, ...]: ...
    def get_snapshot(self) -> Snapshot: ...

class Policy(Protocol):
    def act(self, obs: np.ndarray) -> int: ...
    def act_batch(self, obs_batch: np.ndarray) -> np.ndarray: ...

# from core/feature_iface.py
class Featurizer(Protocol):
    """Encodes a Snapshot into an observation array."""
    def shape(self, grid_h: int, grid_w: int) -> Tuple[int, ...]: ...
    def encode(self, snap: Snapshot) -> np.ndarray: ...

# from core/reward_iface.py
class RewardAdapter(Protocol):
    """Maps (prev, cur) snapshots to a scalar reward."""
    def compute(self, prev: Snapshot, cur: Snapshot) -> float: ...
