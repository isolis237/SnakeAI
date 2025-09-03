from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters and knobs for DQN-style algorithms."""
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 100_000
    min_replay_before_learn: int = 10_000
    learn_every: int = 1                 # env steps per learner step (1 = every step)
    target_update: Literal["hard", "soft"] = "hard"
    target_update_interval: int = 1000   # for hard updates
    target_soft_tau: float = 0.005       # for soft/Polyak
    max_grad_norm: Optional[float] = 10.0
    double_dqn: bool = True
    dueling: bool = False
    prioritized_replay: bool = False     # interface placeholder
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 250_000
    device: str = "auto"
    seed: Optional[int] = None
