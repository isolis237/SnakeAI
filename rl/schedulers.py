from __future__ import annotations
from typing import Protocol

class EpsilonScheduler(Protocol):
    def value(self, global_step: int) -> float: ...

class LinearDecayEpsilon(EpsilonScheduler):
    """Simple working default. Feel free to swap later."""
    def __init__(self, start: float, end: float, steps: int):
        self.start = start; self.end = end; self.steps = max(1, steps)
    def value(self, global_step: int) -> float:
        t = min(global_step, self.steps)
        return self.start + (self.end - self.start) * (t / self.steps)
