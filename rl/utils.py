# snake/rl/utils.py
from __future__ import annotations
from typing import Protocol
import torch

def resolve_device(pref: str | None = "auto") -> str:
    """
    Returns 'cuda' if available and pref is 'auto', else 'mps' (Apple) if available,
    otherwise 'cpu'. If pref is a concrete device string, returns it unchanged.
    """
    if pref is None or pref.lower() == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # mac
            return "mps"
        return "cpu"
    return pref

DIRS = [(1,0),(0,1),(-1,0),(0,-1)]
def dir_to_abs(cur_dir): 
    return DIRS.index(cur_dir)


class EpsilonScheduler(Protocol):
    def value(self, global_step: int) -> float: ...

class LinearDecayEpsilon(EpsilonScheduler):
    """Simple working default. Feel free to swap later."""
    def __init__(self, start: float, end: float, steps: int):
        self.start = start; self.end = end; self.steps = max(1, steps)
    def value(self, global_step: int) -> float:
        t = min(global_step, self.steps)
        return self.start + (self.end - self.start) * (t / self.steps)

