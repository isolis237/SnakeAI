from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

from .replay import Transition, ReplayBuffer


@dataclass
class _Slot:
    obs: Optional[np.ndarray] = None
    action: Optional[int] = None
    reward: Optional[float] = None
    next_obs: Optional[np.ndarray] = None
    terminated: Optional[bool] = None
    truncated: Optional[bool] = None
    info: Optional[dict] = None


class RingBuffer(ReplayBuffer):
    """Simple uniform replay buffer (numpy-backed ring).

    - Stores python/numpy objects per slot; efficient enough for Snake.
    - `get_state()/set_state()` keep indices/size only (lightweight checkpoint). The
      actual contents are *not* serialized by default.
    """

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        self._cap = int(capacity)
        self._buf: List[_Slot] = [_Slot() for _ in range(self._cap)]
        self._size = 0
        self._head = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def capacity(self) -> int:
        return self._cap

    def add(self, t: Transition) -> None:
        i = self._head
        self._buf[i] = _Slot(
            obs=t.obs,
            action=int(t.action),
            reward=float(t.reward),
            next_obs=t.next_obs,
            terminated=bool(t.terminated),
            truncated=bool(t.truncated),
            info=dict(t.info) if t.info is not None else {},
        )
        self._head = (self._head + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, batch_size: int) -> Tuple[Transition, ...]:
        assert self._size > 0, "Cannot sample from empty buffer"
        idx = self._rng.integers(self._size, size=int(batch_size))
        out = []
        for i in idx:
            s = self._buf[i]
            out.append(Transition(
                obs=s.obs, action=s.action, reward=s.reward, next_obs=s.next_obs,
                terminated=s.terminated, truncated=s.truncated, info=s.info,
            ))
        return tuple(out)

    # --- lightweight checkpoint (indices only) ---
    def get_state(self) -> dict:
        return {
            "size": self._size,
            "head": self._head,
            "capacity": self._cap,
            "rng_state": self._rng.bit_generator.state,
        }

    def set_state(self, state: dict) -> None:
        assert state["capacity"] == self._cap, "Capacity mismatch when restoring RingBuffer"
        self._size = int(state["size"])
        self._head = int(state["head"])
        self._rng.bit_generator.state = state["rng_state"]