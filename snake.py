# snake.py

import pygame as pg
import numpy as np
from collections import deque
import random

class SnakeGame:
    def __init__(self, grid_w=20, grid_h=15, cell=32, fps=10, seed=None):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell = cell
        self.fps = fps
        self.rng = random.Random(seed)
        self.screen = None
        self.clock = None
        self.running = False
        self.reset()

    def init(self, title="Snake", surface=None):
        if surface is None:
            pg.init()
            pg.display.set_caption(title)
            self.screen = pg.display.set_mode((self.grid_w * self.cell, self.grid_h * self.cell))
            self.clock = pg.time.Clock()
            self.running = True
        else:
            if not pg.get_init():
                pg.init()
            self.screen = surface
            self.clock = None
            self.running = True

    def reset(self):
        cx, cy = self.grid_w // 2, self.grid_h // 2
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.dir = (1, 0)
        self.just_ate = False
        self.steps_since_food = 0
        self.score = 0
        self.done = False
        self.food = self._spawn_food()
        return self._obs()

    def step(self, action=None):
        if self.done:
            return self._obs(), 0.0, True, {"reason": "game_over"}

        # Only poll events if this game owns the window (single-game mode)
        if self.clock is not None:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                    self.done = True
                    return self._obs(), 0.0, True, {"reason": "quit"}
                elif event.type == pg.KEYDOWN and action is None:
                    if event.key == pg.K_UP and self.dir != (0, 1):   self.dir = (0, -1)
                    elif event.key == pg.K_DOWN and self.dir != (0, -1): self.dir = (0, 1)
                    elif event.key == pg.K_LEFT and self.dir != (1, 0):  self.dir = (-1, 0)
                    elif event.key == pg.K_RIGHT and self.dir != (-1, 0): self.dir = (1, 0)
                    elif event.key == pg.K_ESCAPE:
                        self.running = False
                        self.done = True
                        return self._obs(), 0.0, True, {"reason": "quit"}

        if action is not None:
            self.dir = self._apply_relative_turn(self.dir, action)

        head_x, head_y = self.snake[0]
        dx, dy = self.dir
        new_head = (head_x + dx, head_y + dy)

        reward = 0.0
        self.steps_since_food += 1

        if not (0 <= new_head[0] < self.grid_w and 0 <= new_head[1] < self.grid_h):
            self.done = True
            reward = -1.0
        elif new_head in self.snake:
            self.done = True
            reward = -1.0
        else:
            self.snake.appendleft(new_head)
            if new_head == self.food:
                self.score += 1
                reward = +1.0
                self.food = self._spawn_food()
                self.just_ate = True
                self.steps_since_food = 0
            else:
                self.snake.pop()
                self.just_ate = False

        if not self.done and self.steps_since_food > self.grid_w * self.grid_h * 2:
            self.done = True
            reward = -0.5

        if self.screen:
            self._render()  # draw snake/food on our surface

        if self.clock:
            self.clock.tick(self.fps)

        return self._obs(), reward, self.done, {"score": self.score}

    def close(self):
        pg.quit()

    # ---------- helpers ----------
    def _spawn_food(self):
        free = {(x, y) for x in range(self.grid_w) for y in range(self.grid_h)} - set(self.snake)
        return self.rng.choice(list(free))

    def _apply_relative_turn(self, cur_dir, action):
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # R,D,L,U
        i = dirs.index(cur_dir)
        if action == 0: ni = i
        elif action == 1: ni = (i - 1) % 4
        elif action == 2: ni = (i + 1) % 4
        else: ni = i
        return dirs[ni]

    def _render(self):
        c = self.cell
        surf = self.screen
        surf.fill((15, 15, 15))
        fx, fy = self.food
        pg.draw.rect(surf, (220, 70, 70), pg.Rect(fx * c, fy * c, c, c))
        for idx, (x, y) in enumerate(self.snake):
            color = (60, 200, 90) if idx == 0 else (40, 160, 70)
            pg.draw.rect(surf, color, pg.Rect(x * c, y * c, c, c))

    def draw_overlay(self, font, is_leader=False, show_score=True):
        """Draw per-tile overlay (score + border)."""
        surf = self.screen
        # Score text (top-right)
        if show_score:
            txt = font.render(f"Score: {self.score}", True, (230, 230, 230))
            tr = txt.get_rect()
            tr.topright = (surf.get_width() - 6, 4)
            surf.blit(txt, tr)

        # Border: yellow if leader, dark otherwise
        border_color = (235, 210, 60) if is_leader else (35, 35, 35)
        pg.draw.rect(surf, border_color, surf.get_rect(), width=2)

    # observations (unchanged)
    def _obs(self):
        obs = np.zeros((self.grid_h, self.grid_w, 3), dtype=np.float32)
        for (x, y) in list(self.snake)[1:]:
            obs[y, x, 0] = 1.0
        hx, hy = self.snake[0]
        obs[hy, hx, 1] = 1.0
        fx, fy = self.food
        obs[fy, fx, 2] = 1.0
        return obs

    def _obs_visual(self):
        grid = [["." for _ in range(self.grid_w)] for _ in range(self.grid_h)]
        fx, fy = self.food
        grid[fy][fx] = "F"
        for (x, y) in list(self.snake)[1:]:
            grid[y][x] = "o"
        hx, hy = self.snake[0]
        grid[hy][hx] = "H"
        return ["".join(row) for row in grid]
