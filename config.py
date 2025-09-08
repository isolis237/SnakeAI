from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, List

@dataclass
class EnvConfig:
    grid_w: int = 12
    grid_h: int = 12
    n_actions: int = 3
    max_ep_steps: int = 0

@dataclass
class RenderConfig:
    cell_px: int = 24
    title: str = "Snake"
    grid_lines: bool = False
    show_hud: bool = True
    record_dir: Optional[str] = None
    # from main.py
    render: bool = False
    live_view: bool = False
    view_fps: int = 12
    hide_hud: bool = False # maps to show_hud

@dataclass
class DQNConfig:
    """Hyperparameters and knobs for DQN-style algorithms."""
    gamma: float = 0.955
    lr: float = 1e-3 # from main.py
    batch_size: int = 64 # from main.py
    replay_capacity: int = 150_000 # from main.py cap
    min_replay_before_learn: int = 8_000 # from main.py warmup
    learn_every: int = 1
    target_update: Literal["hard", "soft"] = "soft"
    target_update_interval: int = 1000
    target_soft_tau: float = 0.005
    max_grad_norm: Optional[float] = 10.0
    double_dqn: bool = True
    dueling: bool = False
    prioritized_replay: bool = False
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 85_000
    device: str = "auto"
    seed: Optional[int] = 0 # from main.py

@dataclass
class RunnerConfig:
    mode: str = "snake"
    episodes: int = 50
    fps: int = 8
    n_envs: int = 8
    episodes_per_env: int = 3
    log_every_updates: int = 100
    echo_every_steps: int = 1_500
    view_every_updates: int = 500
    stream_ep_prob: float = 0.010
    best_metric: str = "reward"
    viewer_seed: int = 0
    player_buffer: int = 4
    player_gap: float = 0.35
    flat_dim: Optional[int] = None

@dataclass
class AppConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
