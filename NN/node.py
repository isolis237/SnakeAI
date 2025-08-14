import pygame as pg
import math
import random
import numpy

class Node(pg.sprite.Sprite):
    def __init__(self, pos, layer=0, inputs=None, r=12, color=(220,220,220), rect_aspect=(1,1),
                 rect_color=(100,100,100), rect_margin_px=0, rect_border_radius=0):
        super().__init__()
        self.layer = layer
        self.value = value
        self.inputs = inputs

        self.r = r
        self._color = pg.Color(color)
        self.image = pg.Surface((2*r, 2*r), pg.SRCALPHA)

        # draw Node
        pg.draw.circle(self.image, color, (r, r), r)

        # --- draw an inscribed rectangle (axis-aligned) ---
        aw, ah = rect_aspect

        # scale to fit: corner of rect lies on circle of radius (r - margin)
        effective_r = max(0, r - rect_margin_px)
        k = (2 * effective_r) / math.sqrt(aw*aw + ah*ah) if (aw and ah) else 0
        W = int(round(k * aw))
        H = int(round(k * ah))
        # top-left so it's centered in the circle
        x = r - W // 2
        y = r - H // 2
        # width=0 means filled; change to >0 for outline thickness
        pg.draw.rect(self.image, rect_color, pg.Rect(x, y, W, H), width=0, border_radius=rect_border_radius)

        # position sprite on screen
        self.rect = self.image.get_rect(center=pos)

        # render weight inside the rectangle
        font = pg.font.Font(None, max(12, int(min(W, H)*0.6)))
        surf = font.render(str(weight), True, (20, 20, 20))
        self.image.blit(surf, surf.get_rect(center=(r, r)))

    @property
    def x(self): return self.rect.x
    @x.setter
    def x(self, v): self.rect.x = int(v)

    @property
    def y(self): return self.rect.y
    @y.setter
    def y(self, v): self.rect.y = int(v)
    
    def draw(self, surface):
        surface.blit(self.image, self.rect)