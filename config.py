# config.py
from dataclasses import dataclass, replace
from typing import Optional

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