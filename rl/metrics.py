from __future__ import annotations
from collections import deque
from typing import Deque, Dict, Any, Optional

class EMA:
    """Exponential moving average."""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Optional[float] = None
    def update(self, x: float) -> float:
        self.value = x if self.value is None else (self.alpha * x + (1 - self.alpha) * self.value)
        return self.value

class WindowedStat:
    """Fixed-window mean/min/max."""
    def __init__(self, window: int):
        self.window = window
        self.buf: Deque[float] = deque(maxlen=window)
    def add(self, x: float) -> None:
        self.buf.append(float(x))
    def summary(self) -> Dict[str, float]:
        if not self.buf:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        b = list(self.buf)
        return {"mean": sum(b) / len(b), "min": min(b), "max": max(b)}
