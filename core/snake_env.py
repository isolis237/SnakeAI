# snake/core/snake_env.py
from __future__ import annotations
from collections import deque
import numpy as np
from .interfaces import Env, StepResult, Snapshot
from .snake_rules import Rules, Config

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

class SnakeEnv(Env):
    def __init__(
        self,
        rules: Rules,
        obs_mode: str = "ch3",
        reward_mode: str = "basic",
        include_walls: bool = True,
        include_dir: bool = True,
    ):
        self.rules = rules
        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.include_walls = include_walls
        self.include_dir = include_dir

        # --- shaping state ---
        self._recent_heads = deque(maxlen=8)   # loop/oscillation detection
        self._prev_food_dist: float | None = None
        self._steps_since_bite: int = 0        # track steps since last bite

        # --- reward knobs (replace your current values with these) ---
        self._living_penalty      = -0.003   # small “doing nothing” penalty
        self._dist_gain_weight    =  0.035   # distance-to-food base weight
        self._dist_gain_scale     =  0.30   # stronger pull as snake grows

        self._bite_base           =  2.05    # clear positive signal for eating
        self._bite_gain_per_pt    =  0.25    # later bites worth more, but not explosive
        self._survive_per_step    =  0.002   # scaled by score each step

        self._terminal_score_w    =  0.05    # dying with higher score hurts less
        self._loop_penalty        = -0.020   # penalize tiny orbits

        # terminal reason extras (on top of -1.0 base)
        self._wall_death_extra       = -0.60
        self._self_death_extra       = -0.45
        self._starvation_death_extra = -0.25

        # post-bite “clean movement” boost (decays over a few steps)
        self._post_bite_bonus_steps = 8
        self._post_bite_step_bonus  = 0.010


    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rules.seed(seed)
        snap = self.rules.reset()
        self._recent_heads.clear()
        self._recent_heads.append(snap.snake[0])
        self._prev_food_dist = manhattan(snap.snake[0], snap.food)
        self._steps_since_bite = 0
        return self._encode_obs(snap)

    def step(self, action: int) -> StepResult:
        prev = self.rules.snapshot()
        snap = self.rules.step(action)

        # compute reward BEFORE updating trackers
        reward = self._reward(prev, snap)
        obs = self._encode_obs(snap)

        # update trackers AFTER reward (so deltas use prev state)
        self._recent_heads.append(snap.snake[0])
        self._prev_food_dist = manhattan(snap.snake[0], snap.food)

        # update post-bite counter
        if snap.score > prev.score:
            self._steps_since_bite = 0
        else:
            self._steps_since_bite += 1

        return StepResult(
            obs=obs, reward=reward,
            terminated=snap.terminated, truncated=False,
            info={"snapshot": snap, "score": snap.score, "reason": snap.reason}
        )

    def action_space_n(self) -> int:
        return 3 if self.rules.cfg.relative_actions else 4  # relative: straight/left/right (or 4 if absolute)

    def observation_shape(self) -> tuple[int, ...]:
        H = self.rules.cfg.grid_h
        W = self.rules.cfg.grid_w
        C = 3 + (1 if self.include_walls else 0) + (2 if self.include_dir else 0)
        if self.obs_mode == "ch3":
            return (H, W, C)
        return (H * W * C,)

    def get_snapshot(self) -> Snapshot:
        return self.rules.snapshot()

    # --- helpers ---
    def _encode_obs(self, s: Snapshot) -> np.ndarray:
        H, W = s.grid_h, s.grid_w
        C = 3 + (1 if self.include_walls else 0) + (2 if self.include_dir else 0)
        grid = np.zeros((H, W, C), dtype=np.float32)

        ch = 0
        # body (excluding head)
        for (x, y) in list(s.snake)[1:]:
            grid[y, x, ch] = 1.0
        ch += 1

        # head
        hx, hy = s.snake[0]
        grid[hy, hx, ch] = 1.0
        ch += 1

        # food
        fx, fy = s.food
        grid[fy, fx, ch] = 1.0
        ch += 1

        # walls (borders = 1)
        if self.include_walls:
            grid[0, :, ch] = 1.0
            grid[H - 1, :, ch] = 1.0
            grid[:, 0, ch] = 1.0
            grid[:, W - 1, ch] = 1.0
            ch += 1

        # direction channels: broadcast (dx, dy)
        if self.include_dir:
            dx, dy = s.dir  # e.g., (-1,0),(1,0),(0,-1),(0,1)
            grid[:, :, ch] = float(dx); ch += 1
            grid[:, :, ch] = float(dy); ch += 1

        if self.obs_mode == "flat":
            return grid.reshape(-1)
        return grid

    def _reward(self, prev: Snapshot, cur: Snapshot) -> float:
        # --- terminal: base penalty + reason-specific extra + credit for achieved score ---
        if cur.terminated:
            extra = 0.0
            # cur.reason typically in {'wall','self','starvation','manual'}
            if cur.reason == 'wall':
                extra += self._wall_death_extra
            elif cur.reason == 'self':
                extra += self._self_death_extra
            elif cur.reason == 'starvation':
                extra += self._starvation_death_extra
            return -1.0 + extra + self._terminal_score_w * float(prev.score)

        # --- ate food: base + score-scaled bite bonus ---
        if cur.score > prev.score:
            return self._bite_base + self._bite_gain_per_pt * float(cur.score)

        # --- shaped non-terminal step ---
        r = 0.0
        r += self._living_penalty
        r += self._survive_per_step * float(cur.score)

        # distance shaping (score-scaled pull)
        if self._prev_food_dist is not None:
            new_dist = manhattan(cur.snake[0], cur.food)
            delta = float(self._prev_food_dist - new_dist)  # + if closer, - if farther
            dist_w = self._dist_gain_weight + self._dist_gain_scale * float(cur.score)
            r += dist_w * delta

        # loop / oscillation penalty (very small orbit -> penalize)
        if len(self._recent_heads) >= self._recent_heads.maxlen:
            if len(set(self._recent_heads)) <= 3:
                r += self._loop_penalty

        # --- new: post-bite streak encouragement (only if not looping) ---
        # For N steps after eating, give a small decaying bonus if motion isn't looping.
        steps = self._steps_since_bite
        if steps <= self._post_bite_bonus_steps:
            # consider "healthy movement" if we used at least 5 unique cells recently
            unique = len(set(self._recent_heads))
            if unique >= 5:
                # linear decay: big right after bite, fades out
                factor = (self._post_bite_bonus_steps - steps) / float(self._post_bite_bonus_steps)
                r += self._post_bite_step_bonus * factor

        r = max(-1.0, min(1.5, r))
        return r
