# snake/core/snake_rules.py  (pure rules, no pygame)
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import random
import numpy as np
from .interfaces import Snapshot

@dataclass(frozen=True)
class Config:
    grid_w: int = 12
    grid_h: int = 12
    start_len: int = 3
    relative_actions: bool = True
    max_steps_without_food: Optional[int] = None
    seed: Optional[int] = None

class Rules:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._reset_state()

    def seed(self, seed: Optional[int]):
        self.rng = random.Random(seed)

    def _reset_state(self):
        cx, cy = self.cfg.grid_w//2, self.cfg.grid_h//2
        self.snake = [(cx-i, cy) for i in range(self.cfg.start_len)]
        self.dir = (1,0)
        self.food = self._place_food()
        self.score = 0
        self.steps_since_food = 0
        self.step_count = 0
        self.terminated = False
        self.reason = None

    def reset(self) -> Snapshot:
        self._reset_state()
        return self.snapshot()

    def _place_food(self) -> Tuple[int,int]:
        occ = set(self.snake)
        free = [(x,y) for x in range(self.cfg.grid_w) for y in range(self.cfg.grid_h) if (x,y) not in occ]
        return self.rng.choice(free)

    def _apply_relative(self, cur, action):
        dirs = [(1,0),(0,1),(-1,0),(0,-1)]
        i = dirs.index(cur)
        return dirs[{0:i,1:(i-1)%4,2:(i+1)%4}.get(action, i)]

    def step_dir(self, action: int):
        if self.cfg.relative_actions:
            self.dir = self._apply_relative(self.dir, action)
        else:
            # absolute: action directly selects a DIRS entry
            ndx, ndy = [(1,0), (0,1), (-1,0), (0,-1)][action]
            cdx, cdy = self.dir

            # Prevent instant 180° reversal if the snake has a body
            if len(self.snake) > 1 and (ndx == -cdx and ndy == -cdy):
                # ignore invalid reverse; keep current direction
                return

            self.dir = (ndx, ndy)

    def step(self, action: int) -> Snapshot:
        if self.terminated:
            return self.snapshot()
        self.step_dir(action)

        hx, hy = self.snake[0]
        dx, dy = self.dir
        new_head = (hx+dx, hy+dy)
        self.step_count += 1
        self.steps_since_food += 1

        # collisions
        if not (0 <= new_head[0] < self.cfg.grid_w and 0 <= new_head[1] < self.cfg.grid_h):
            self.terminated, self.reason = True, "wall"
            return self.snapshot()
        if new_head in self.snake:
            self.terminated, self.reason = True, "self"
            return self.snapshot()

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            self.steps_since_food = 0
        else:
            self.snake.pop()

        if (self.cfg.max_steps_without_food is not None and
            self.steps_since_food > self.cfg.max_steps_without_food):
            self.terminated, self.reason = True, "starvation"
        return self.snapshot()

    def snapshot(self) -> Snapshot:
        return Snapshot(
            snake=tuple(self.snake),
            food=self.food,
            dir=self.dir,
            score=self.score,
            steps_since_food=self.steps_since_food,
            step_count=self.step_count,
            terminated=self.terminated,
            reason=self.reason,
            grid_w=self.cfg.grid_w,
            grid_h=self.cfg.grid_h,
        )
