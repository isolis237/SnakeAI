from __future__ import annotations
from typing import Protocol
from .interfaces import Snapshot

class RewardAdapter(Protocol):
    """Maps (prev, cur) snapshots to a scalar reward."""
    def compute(self, prev: Snapshot, cur: Snapshot) -> float: ...
