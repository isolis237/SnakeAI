# snake/viz/keyboard.py
import pygame as pg

class Keyboard:
    def __init__(self, relative=True):
        self.relative = relative

    def poll(self):
        for e in pg.event.get():
            if e.type == pg.QUIT:
                return "quit"
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_ESCAPE: return "quit"
                if self.relative:
                    if e.key == pg.K_LEFT:  return 1
                    if e.key == pg.K_RIGHT: return 2
                    if e.key == pg.K_UP:    return 0  # straight
                else:
                    if e.key == pg.K_RIGHT: return 0
                    if e.key == pg.K_DOWN:  return 1
                    if e.key == pg.K_LEFT:  return 2
                    if e.key == pg.K_UP:    return 3
        return None
