from __future__ import annotations
from .interfaces import Snapshot
from typing import Protocol

class RewardAdapter(Protocol):
    """Maps (prev, cur) snapshots to a scalar reward."""
    def compute(self, prev: Snapshot, cur: Snapshot) -> float: ...

class BasicShaping(RewardAdapter):
    def __init__(self):
        # copy your knob defaults here if you want
        pass
    def compute(self, prev: Snapshot, cur: Snapshot) -> float:
        raise NotImplementedError
