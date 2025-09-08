# snake/viz/renderer_headless.py
from __future__ import annotations
from interfaces import Snapshot
from interfaces.render import RenderConfig, Renderer

class HeadlessRenderer(Renderer):
    def open(self, grid_w: int, grid_h: int, cfg: RenderConfig) -> None:
        self.cfg = cfg
        self.w = grid_w
        self.h = grid_h
    def draw(self, snap: Snapshot) -> None:
        pass  # no-op
    def tick(self, fps: int) -> None:
        pass
    def close(self) -> None:
        pass
    def save_frame(self, snap: Snapshot) -> None:
        pass
