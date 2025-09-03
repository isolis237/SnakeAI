# snake/viz/render_iface.py
from __future__ import annotations
from typing import Protocol, Optional
from core.interfaces import Snapshot

class RenderConfig:
    def __init__(
        self,
        cell_px: int = 24,
        title: str = "Snake",
        grid_lines: bool = False,
        show_hud: bool = True,
        record_dir: Optional[str] = None,   # if set, save frames here as PNGs
    ):
        self.cell_px = cell_px
        self.title = title
        self.grid_lines = grid_lines
        self.show_hud = show_hud
        self.record_dir = record_dir

class Renderer(Protocol):
    def open(self, grid_w: int, grid_h: int, cfg: RenderConfig) -> None: ...
    def draw(self, snap: Snapshot) -> None: ...
    def tick(self, fps: int) -> None: ...
    def close(self) -> None: ...
    def save_frame(self, snap: Snapshot) -> None: ...
