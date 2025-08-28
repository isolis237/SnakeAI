# snake/core/snake_env.py (RL adapter; still no pygame/torch)
from __future__ import annotations
import numpy as np
from .interfaces import Env, StepResult, Snapshot
from .snake_rules import Rules, Config

class SnakeEnv(Env):
    def __init__(self, rules: Rules, obs_mode: str = "ch3", reward_mode: str = "basic"):
        self.rules = rules
        self.obs_mode = obs_mode
        self.reward_mode = reward_mode

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rules.seed(seed)
        snap = self.rules.reset()
        return self._encode_obs(snap)

    def step(self, action: int) -> StepResult:
        prev = self.rules.snapshot()
        snap = self.rules.step(action)
        reward = self._reward(prev, snap)
        obs = self._encode_obs(snap)
        return StepResult(
            obs=obs, reward=reward,
            terminated=snap.terminated, truncated=False,
            info={"snapshot": snap, "score": snap.score, "reason": snap.reason}
        )

    def action_space_n(self) -> int:
        return 3  # relative: straight/left/right (or 4 if absolute)

    def observation_shape(self) -> tuple[int, ...]:
        if self.obs_mode == "ch3":
            return (self.rules.cfg.grid_h, self.rules.cfg.grid_w, 3)
        return (self.rules.cfg.grid_h * self.rules.cfg.grid_w * 3,)

    def get_snapshot(self) -> Snapshot:
        return self.rules.snapshot()

    # --- helpers ---
    def _encode_obs(self, s: Snapshot) -> np.ndarray:
        H, W = s.grid_h, s.grid_w
        grid = np.zeros((H, W, 3), dtype=np.float32)
        for (x,y) in list(s.snake)[1:]:
            grid[y,x,0] = 1.0
        hx, hy = s.snake[0]
        grid[hy,hx,1] = 1.0
        fx, fy = s.food
        grid[fy,fx,2] = 1.0
        if self.obs_mode == "flat":
            return grid.reshape(-1)
        return grid

    def _reward(self, prev: Snapshot, cur: Snapshot) -> float:
        if cur.terminated:
            return -1.0
        if cur.score > prev.score:
            return +1.0
        return 0.0  # keep simple; you can add -step shaping later
