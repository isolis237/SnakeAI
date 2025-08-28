# snake/core/interfaces.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Protocol
import numpy as np

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
