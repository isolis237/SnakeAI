# snake/viz/renderer_pygame.py
from __future__ import annotations
import os
import pygame as pg
from typing import Optional
from core.interfaces import Snapshot
import viz.renderer_colors as theme  # keep your existing module
from .render_iface import Renderer, RenderConfig

class PygameRenderer(Renderer):
    def __init__(self):
        self.cell = 24
        self.surf: Optional[pg.Surface] = None
        self.clock: Optional[pg.time.Clock] = None
        self._auto_flip = True
        self._grid_w = 0
        self._grid_h = 0
        self._cfg = RenderConfig()
        self._frame_idx = 0

    # --- Protocol: open / draw / tick / close / save_frame ---
    def open(self, grid_w: int, grid_h: int, cfg: RenderConfig) -> None:
        self._grid_w, self._grid_h = grid_w, grid_h
        self._cfg = cfg
        self.cell = cfg.cell_px

        pg.init()
        pg.display.set_caption(cfg.title)
        self.surf = pg.display.set_mode((grid_w * self.cell, grid_h * self.cell))
        self.clock = pg.time.Clock()
        self._auto_flip = True
        self._frame_idx = 0

        if cfg.record_dir:
            os.makedirs(cfg.record_dir, exist_ok=True)

    def draw(self, s: Snapshot) -> None:
        assert self.surf is not None, "Renderer not opened"
        surf = self.surf
        c = self.cell

        # basic event pump so the window stays responsive
        for event in pg.event.get():
            if event.type == pg.QUIT:
                # don’t quit process here; caller controls loop
                pass

        # background
        surf.fill(theme.BG)

        # optional gridlines
        if self._cfg.grid_lines:
            for x in range(0, self._grid_w * c, c):
                pg.draw.line(surf, theme.GRID, (x, 0), (x, self._grid_h * c))
            for y in range(0, self._grid_h * c, c):
                pg.draw.line(surf, theme.GRID, (0, y), (self._grid_w * c, y))

        # food
        fx, fy = s.food
        pg.draw.rect(surf, theme.FOOD, pg.Rect(fx * c, fy * c, c, c))

        # snake (head, then body)
        for i, (x, y) in enumerate(s.snake):
            col = theme.HEAD if i == 0 else theme.BODY
            pg.draw.rect(surf, col, pg.Rect(x * c, y * c, c, c))

        # HUD
        if self._cfg.show_hud:
            font = pg.font.SysFont(None, 22)
            txt = font.render(
                f"Score: {s.score}   Steps: {s.step_count}   Dir: {s.dir}   Reason: {s.reason or ''}",
                True, theme.TEXT
            )
            surf.blit(txt, (6, 4))

        if self._auto_flip:
            pg.display.flip()

        # optional recording
        if self._cfg.record_dir:
            self._save_surface_frame()

    def tick(self, fps: int) -> None:
        if self.clock:
            self.clock.tick(fps)

    def close(self) -> None:
        try:
            pg.quit()
        except Exception:
            pass
        self.surf = None
        self.clock = None

    def save_frame(self, s: Snapshot) -> None:
        """Force-save a frame of the *current* surface contents (after draw)."""
        if not self._cfg.record_dir or self.surf is None:
            return
        self._save_surface_frame()

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

    # --- internals ---
    def _save_surface_frame(self) -> None:
        assert self.surf is not None
        fname = os.path.join(self._cfg.record_dir, f"frame_{self._frame_idx:06d}.png")
        pg.image.save(self.surf, fname)
        self._frame_idx += 1
