# viz/renderer_pygame.py
from __future__ import annotations
import os
from pathlib import Path
import pygame as pg
from typing import Optional, Union
from config import AppConfig
from core.interfaces import Snapshot
import viz.renderer_colors as theme

PathLike = Union[str, bytes, os.PathLike]

class PygameRenderer:
    def __init__(self):
        self.cell = 24
        self.cfg: Optional[AppConfig] = None
        self.surf: Optional[pg.Surface] = None
        self.clock: Optional[pg.time.Clock] = None
        self._auto_flip = True
        self._grid_w = 0
        self._grid_h = 0
        self._frame_idx = 0
        self._overlay_text: Optional[str] = None

    def set_overlay(self, text: Optional[str]) -> None:
        self._overlay_text = text or ""

    def open(self, cfg: AppConfig) -> None:
        # Guard: ensure instance, not class
        if isinstance(cfg, type):
            raise TypeError("Pass an AppConfig instance (use AppConfig()), not the class.")
        self.cfg = cfg  

        self._grid_w, self._grid_h = cfg.grid_w, cfg.grid_h
        self.cell = cfg.render_cell

        pg.init()
        pg.display.set_caption(cfg.render_title)
        self.surf = pg.display.set_mode((self._grid_w * self.cell, self._grid_h * self.cell))
        self.clock = pg.time.Clock()
        self._auto_flip = True
        self._frame_idx = 0

        if cfg.render_record_dir:
            os.makedirs(cfg.render_record_dir, exist_ok=True)

    def draw(self, s: Snapshot) -> None:
        assert self.surf is not None, "Renderer not opened"
        assert self.cfg is not None, "Renderer config not set (call open first)"
        surf = self.surf
        c = self.cell

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pass

        surf.fill(theme.BG)

        fx, fy = s.food
        pg.draw.rect(surf, theme.FOOD, pg.Rect(fx * c, fy * c, c, c))

        for i, (x, y) in enumerate(s.snake):
            col = theme.HEAD if i == 0 else theme.BODY
            pg.draw.rect(surf, col, pg.Rect(x * c, y * c, c, c))

        if self.cfg.render_show_hud:
            font = pg.font.SysFont(None, 22)
            txt = font.render(
                f"Score: {s.score}   Steps: {s.step_count}   Dir: {s.dir}   Reason: {s.reason or ''}",
                True, theme.TEXT
            )
            surf.blit(txt, (6, 4))

        if self._overlay_text:
            font = pg.font.SysFont(None, 22)
            ovr = font.render(self._overlay_text, True, theme.TEXT)
            surf.blit(ovr, (6, 26))

        if self._auto_flip:
            pg.display.flip()

        if self.cfg.render_record_dir:
            self._save_surface_frame()

    def tick(self, fps: int) -> None:
        if self.clock:
            self.clock.tick(fps)

    def close(self) -> None:
        try:
            pg.quit()
        finally:
            self.surf = None
            self.clock = None

    def save_frame(self, s: Snapshot) -> None:
        assert self.cfg is not None, "Renderer config not set (call open first)"
        if not self.cfg.render_record_dir or self.surf is None:
            return
        self._save_surface_frame()

    # internals
    def _save_surface_frame(self) -> None:
        assert self.surf is not None
        assert self.cfg is not None
        rec_dir: PathLike = self.cfg.render_record_dir
        if not isinstance(rec_dir, (str, bytes, os.PathLike)):
            raise TypeError(f"render_record_dir must be path-like, got {type(rec_dir)}")
        fname = os.path.join(rec_dir, f"frame_{self._frame_idx:06d}.png")
        pg.image.save(self.surf, fname)
        self._frame_idx += 1

    # --- Backward-compat helpers (optional) ---
    def create_window(self, w: int, h: int, title: str = "Snake") -> None:
        """Backwards-compatible wrapper for your old API."""
        self.open(w, h, RenderConfig(cell_px=self.cell, title=title))

    def attach_surface(self, surface: pg.Surface) -> None:
        if not pg.get_init():
            pg.init()
        self.surf = surface
        self.clock = None  # embedding surface typically controls timing
        self._auto_flip = False
