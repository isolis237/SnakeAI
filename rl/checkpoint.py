# rl/checkpoint.py
from __future__ import annotations
import os, math, json
from typing import Optional, Dict, Any, Protocol
from .dqn_agent import DQNAgent


class Checkpointable(Protocol):
    """Objects that can round-trip their state as pure-Python/JSON-serializable dicts."""
    def get_state(self) -> Dict[str, Any]: ...
    def set_state(self, state: Dict[str, Any]) -> None: ...

class CheckpointManager:
    """Saves/loads a named bundle of components. Each component must be Checkpointable."""
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def save(self, tag: str, components: Dict[str, Checkpointable]) -> None:
        path = os.path.join(self.root_dir, f"{tag}.ckpt.json")
        bundle = {name: comp.get_state() for name, comp in components.items()}
        os.makedirs(self.root_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(bundle, f)

    def load(self, tag: str, components: Dict[str, Checkpointable]) -> None:
        path = os.path.join(self.root_dir, f"{tag}.ckpt.json")
        with open(path, "r") as f:
            bundle = json.load(f)
        for name, comp in components.items():
            if name in bundle:
                comp.set_state(bundle[name])

class Checkpointer:
    """
    Saves model periodically and/or when a 'best' metric improves.
    """
    def __init__(
        self,
        agent: DQNAgent,
        out_dir: str = "runs/snake_dqn",
        tag: str = "dqn",
        save_every_episodes: Optional[int] = None,
        best_metric_key: Optional[str] = "epis/reward_mean100",
    ):
        self.agent = agent
        self.out_dir = out_dir
        self.tag = tag
        self.save_every_episodes = save_every_episodes
        self.best_metric_key = best_metric_key
        self._best = -math.inf

        os.makedirs(out_dir, exist_ok=True)

    def maybe_save_periodic(self, episode_idx: int) -> None:
        if self.save_every_episodes and (episode_idx % self.save_every_episodes == 0):
            self.agent.save(self.out_dir, f"{self.tag}_ep{episode_idx}")

    def maybe_save_best(self, latest_scalars: Dict[str, Any]) -> None:
        if not self.best_metric_key:
            return
        val = latest_scalars.get(self.best_metric_key)
        if val is None:
            return
        try:
            val = float(val)
        except Exception:
            return
        if val > self._best:
            self._best = val
            self.agent.save(self.out_dir, f"{self.tag}_best")

    def save_final(self) -> None:
        self.agent.save(self.out_dir, f"{self.tag}_final")
