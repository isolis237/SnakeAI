# config.py
from dataclasses import dataclass, replace
from typing import Optional, Literal

@dataclass(frozen=True, slots=True)
class AppConfig:
    # shared / global
    grid_w: int = 16
    grid_h: int = 16
    relative_actions: bool = False
    seed: Optional[int] = None

    # gameplay / manual controls
    fps: int = 10
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
    best_metric = "score"

    # MLPQNet
    episodes = 1000
    max_ep_steps = 350
    gamma: float = 0.955
    lr: float = 9.0e-4
    batch_size: int = 128
    replay_capacity: int = 100_000
    min_replay_before_learn = 8_500
    warmup: int = 10_000
    learn_every: int = 1                 # env steps per learner step (1 = every step)
    target_update: Literal["hard", "soft"] = "soft"
    target_update_interval: int = 1000   # for hard updates
    target_soft_tau: float = 0.005       # for soft/Polyak
    max_grad_norm: Optional[float] = 10.0
    double_dqn: bool = True
    dueling: bool = False
    prioritized_replay: bool = False     # interface placeholder
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 85_000
    device: str = "cuda"
    seed: Optional[int] = None


    def with_(self, **kwargs) -> "AppConfig":
        """Convenience: clone with updated values"""
        return replace(self, **kwargs)