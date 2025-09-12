# config.py
from dataclasses import dataclass, replace
from typing import Optional, Literal

@dataclass(frozen=True, slots=True)
class AppConfig:
    # shared / global
    grid_w: int = 10
    grid_h: int = 10
    relative_actions: bool = False
    seed: Optional[int] = None

    # gameplay / manual controls
    fps: int = 24
    start_len: int = 3
    max_steps_without_food: Optional[int] = None

    # render
    render_cell: int = 48
    render_title: str = "Snake"
    render_grid_lines: bool = False
    render_show_hud: bool = True
    render_record_dir: Optional[str] = None

    # live_view
    live_view = True
    max_episodes_buffered = 4
    queue_max = 4
    stream_ep_prob = .01
    best_metric = "reward"

    # MLPQNet
    episodes = 20000
    max_ep_steps = 400
    max_steps_without_food = 80
    gamma: float = 0.965
    lr: float = 1.5e-4
    batch_size: int = 96
    replay_capacity: int = 100_000
    min_replay_before_learn = 15_000
    warmup: int = 15_000
    learn_every: int = 1                 # env steps per learner step (1 = every step)
    target_update: Literal["hard", "soft"] = "soft"
    target_update_interval: int = 1000   # for hard updates
    target_soft_tau: float = 0.015       # for soft/Polyak
    max_grad_norm: Optional[float] = 10.0
    double_dqn: bool = True
    dueling: bool = False
    prioritized_replay: bool = False     # interface placeholder
    epsilon_start: float = 1.0
    epsilon_end: float = 0.035
    epsilon_decay_steps: int = 125_000
    device: str = "cuda"


    def with_(self, **kwargs) -> "AppConfig":
        """Convenience: clone with updated values"""
        return replace(self, **kwargs)