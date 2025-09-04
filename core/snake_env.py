# snake/core/snake_env.py
from __future__ import annotations
from collections import deque
from typing import Optional, Dict, Any, Tuple
import numpy as np
from .interfaces import Env, StepResult, Snapshot
from .snake_rules import Rules, Config
from .feature_iface import Featurizer
from .reward_iface import RewardAdapter

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
        featurizer: Optional[Featurizer] = None,
        reward_adapter: Optional[RewardAdapter] = None,
        seed: Optional[int] = None,
    ):
        self.rules = rules
        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.include_walls = include_walls
        self.include_dir = include_dir
        self._featurizer = featurizer
        self._reward_adapter = reward_adapter

        # RNG for env-level (distinct from Rules RNG)
        self._rng = np.random.default_rng(seed)

        # --- shaping state (unchanged) ---
        self._recent_heads = deque(maxlen=8)
        self._prev_food_dist: float | None = None
        self._steps_since_bite: int = 0

        # --- reward knobs (kept here if no RewardAdapter is provided) ---
        self._living_penalty      = -0.01
        self._dist_gain_weight    =  0.08
        self._dist_gain_scale     =  0.35
        self._bite_base           =  1.15
        self._bite_gain_per_pt    =  0.38
        self._survive_per_step    =  0.002
        self._terminal_score_w    =  0.15
        self._loop_penalty        = -0.075
        self._wall_death_extra       = -0.45
        self._self_death_extra       = -0.15
        self._starvation_death_extra = -0.2
        self._post_bite_bonus_steps = 2
        self._post_bite_step_bonus  = 0.02

    # ---- Env Protocol ----
    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rules.seed(seed)
            self._rng = np.random.default_rng(seed)
        snap = self.rules.reset()
        self._recent_heads.clear()
        self._recent_heads.append(snap.snake[0])
        self._prev_food_dist = manhattan(snap.snake[0], snap.food)
        self._steps_since_bite = 0
        return self._encode_obs(snap)

    def step(self, action: int) -> StepResult:
        prev = self.rules.snapshot()
        snap = self.rules.step(action)

        # reward first
        reward = self._compute_reward(prev, snap)
        obs = self._encode_obs(snap)

        # trackers
        self._recent_heads.append(snap.snake[0])
        self._prev_food_dist = manhattan(snap.snake[0], snap.food)
        if snap.score > prev.score:
            self._steps_since_bite = 0
        else:
            self._steps_since_bite += 1

        # optional action mask
        mask = self._action_mask()

        info: Dict[str, Any] = {
            "snapshot": snap,
            "score": snap.score,
            "reason": snap.reason,
            "action_mask": mask,  # for policies that support masking
        }
        return StepResult(obs=obs, reward=reward, terminated=snap.terminated, truncated=False, info=info)

    def action_space_n(self) -> int:
        return 3 if self.rules.cfg.relative_actions else 4

    def observation_shape(self) -> tuple[int, ...]:
        H = self.rules.cfg.grid_h
        W = self.rules.cfg.grid_w
        if self._featurizer is not None:
            return self._featurizer.shape(H, W)
        C = 3 + (1 if self.include_walls else 0) + (2 if self.include_dir else 0)
        if self.obs_mode == "ch3":
            return (H, W, C)
        return (H * W * C,)

    def get_snapshot(self) -> Snapshot:
        return self.rules.snapshot()

    # ---- Checkpointing hooks (pure-Python) ----
    def get_state(self) -> Dict[str, Any]:
        """Return env+rules state to reproduce training exactly."""
        return {
            "rules": self.rules.get_state(),
            "recent_heads": list(self._recent_heads),
            "prev_food_dist": self._prev_food_dist,
            "steps_since_bite": self._steps_since_bite,
            "obs_mode": self.obs_mode,
            "reward_mode": self.reward_mode,
            "include_walls": self.include_walls,
            "include_dir": self.include_dir,
            "rng_state": self._rng.bit_generator.state,  # numpy RNG state (dict)
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.rules.set_state(state["rules"])
        self._recent_heads.clear()
        for p in state["recent_heads"]:
            self._recent_heads.append(tuple(p))
        self._prev_food_dist = state["prev_food_dist"]
        self._steps_since_bite = int(state["steps_since_bite"])
        self.obs_mode = state["obs_mode"]
        self.reward_mode = state["reward_mode"]
        self.include_walls = bool(state["include_walls"])
        self.include_dir = bool(state["include_dir"])
        self._rng.bit_generator.state = state["rng_state"]

    # ---- Helpers ----
    def _encode_obs(self, s: Snapshot) -> np.ndarray:
        if self._featurizer is not None:
            return self._featurizer.encode(s)

        H, W = s.grid_h, s.grid_w
        C = 3 + (1 if self.include_walls else 0) + (2 if self.include_dir else 0)
        grid = np.zeros((H, W, C), dtype=np.float32)

        ch = 0
        for (x, y) in list(s.snake)[1:]:
            grid[y, x, ch] = 1.0
        ch += 1
        hx, hy = s.snake[0]; grid[hy, hx, ch] = 1.0; ch += 1
        fx, fy = s.food;     grid[fy, fx, ch] = 1.0; ch += 1

        if self.include_walls:
            grid[0, :, ch] = 1.0; grid[H - 1, :, ch] = 1.0
            grid[:, 0, ch] = 1.0; grid[:, W - 1, ch] = 1.0
            ch += 1

        if self.include_dir:
            dx, dy = s.dir
            grid[:, :, ch] = float(dx); ch += 1
            grid[:, :, ch] = float(dy); ch += 1

        if self.obs_mode == "flat":
            return grid.reshape(-1)
        return grid

    def _compute_reward(self, prev: Snapshot, cur: Snapshot) -> float:
        if self._reward_adapter is not None:
            return self._reward_adapter.compute(prev, cur)
        return self._reward(prev, cur)  # fallback to built-in shaping

    def _action_mask(self) -> np.ndarray:
        """Mask illegal actions (absolute mode): block 180° reversals when len>1.
        For relative mode, all 3 actions are always allowed (mask all ones).
        Returns shape: (action_space_n,) with 1.0 allowed, 0.0 disallowed.
        """
        n = self.action_space_n()
        mask = np.ones((n,), dtype=np.float32)
        if not self.rules.cfg.relative_actions and len(self.rules.snake) > 1:
            cdx, cdy = self.rules.dir
            abs_dirs = [(1,0), (0,1), (-1,0), (0,-1)]
            cur_i = abs_dirs.index((cdx, cdy))
            reverse_i = (cur_i + 2) % 4
            mask[reverse_i] = 0.0
        return mask

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
