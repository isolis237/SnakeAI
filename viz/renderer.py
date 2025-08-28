# snake/viz/renderer.py
import pygame as pg
import viz.renderer_colors as theme
from core.interfaces import Snapshot

class Renderer:
    def __init__(self, cell_px: int = 24):
        self.cell = cell_px
        self.surf = None
        self.clock = None

    def create_window(self, w: int, h: int, title="Snake"):
        pg.init()
        pg.display.set_caption(title)
        self.surf = pg.display.set_mode((w*self.cell, h*self.cell))
        self.clock = pg.time.Clock()

    def attach_surface(self, surface: pg.Surface):
        if not pg.get_init():
            pg.init()
        self.surf = surface
        self.clock = None

    def draw(self, s: Snapshot, score_text=True):
        c = self.cell; surf = self.surf
        surf.fill(theme.BG)
        fx, fy = s.food
        pg.draw.rect(surf, theme.FOOD, pg.Rect(fx*c, fy*c, c, c))
        for i,(x,y) in enumerate(s.snake):
            col = theme.HEAD if i==0 else theme.BODY
            pg.draw.rect(surf, col, pg.Rect(x*c, y*c, c, c))
        if score_text:
            font = pg.font.SysFont(None, 22)
            txt = font.render(f"Score: {s.score}", True, theme.TEXT)
            surf.blit(txt, (6, 4))
        pg.display.flip()

    def tick(self, fps: int):
        if self.clock:
            self.clock.tick(fps)

    def close(self):
        pg.quit()
