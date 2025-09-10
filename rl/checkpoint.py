# rl/checkpoint.py
from __future__ import annotations
import os, math
from typing import Optional, Dict, Any
from .dqn_agent import DQNAgent

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
