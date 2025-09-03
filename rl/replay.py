from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Protocol, Tuple
import numpy as np

@dataclass(frozen=True)
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

class ReplayBuffer(Protocol):
    def __len__(self) -> int: ...
    def capacity(self) -> int: ...
    def add(self, t: Transition) -> None: ...
    def sample(self, batch_size: int) -> Tuple[Transition, ...]: ...
    def get_state(self) -> Dict[str, Any]: ...
    def set_state(self, state: Dict[str, Any]) -> None: ...
