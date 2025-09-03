from __future__ import annotations
from .reward_iface import RewardAdapter
from .interfaces import Snapshot

class BasicShaping(RewardAdapter):
    def __init__(self):
        # copy your knob defaults here if you want
        pass
    def compute(self, prev: Snapshot, cur: Snapshot) -> float:
        raise NotImplementedError
