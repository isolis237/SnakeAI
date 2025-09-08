# snake/core/snake_env.py
from __future__ import annotations
from collections import deque
from typing import Optional, Dict, Any, Tuple
import numpy as np
from interfaces import Env, StepResult, Snapshot
from .snake_rules import Rules, Config
from interfaces import Featurizer, RewardAdapter

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
        self._prev_food_dist = None
        self._steps_since_bite = 0
        self._streak = 0                 # NEW: consecutive apples since last death
        self._T_starve = 200             # adjust ~ grid area; 12x12 → ~144–240

        # knobs (kept tight)
        self._k_potential = 0.25         # potential shaping weight
        self._step_cost   = -0.01        # small negative per step
        self._bite_base   = 1.0
        self._bite_streak = 0.25         # extra per current streak count
        self._post_bite_bonus_steps = 3
        self._post_bite_step_bonus  = 0.03
        self._loop_penalty = -0.10
        self._term_score_w = 0.15
        self._death_wall_extra = -0.20
        self._death_self_extra = -0.25
        self._death_starv_extra = -0.10  # usually death reason marks 'starvation'
        self._clip_lo, self._clip_hi = -1.5, 1.5

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
        self._streak = 0 
        return self._encode_obs(snap)

    def step(self, action: int) -> StepResult:
        prev = self.rules.snapshot()
        snap = self.rules.step(action)

        # reward first
        reward = self._compute_reward(prev, snap)
        obs = self._encode_obs(snap)

        # trackers
        self._recent_heads.append(snap.snake[0])
        cur_dist = manhattan(snap.snake[0], snap.food)
        if snap.score > prev.score:
            self._steps_since_bite = 0
            self._streak += 1             # streak increments on each apple
        else:
            self._steps_since_bite += 1

        self._prev_food_dist = cur_dist

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
        # By default returns (gridH, gridW, Channels) (ch3), else flattens
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

        # Obs shape is (gridH, gridW, channels) i.e. (12x12x6)
        # c1 -> snake body excluding head
        # c2 -> snake head
        # c3 -> food
        # c4 -> walls
        # c5 -> DirX
        # c6 -> DirY

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

    # ---- replace _reward() ----
    def _reward(self, prev: Snapshot, cur: Snapshot) -> float:
        # Terminal
        if cur.terminated:
            extra = 0.0
            if cur.reason == 'wall':        extra += self._death_wall_extra
            elif cur.reason == 'self':      extra += self._death_self_extra
            elif cur.reason == 'starvation':extra += self._death_starv_extra
            # credit prior score a bit so dying later while scoring is better than dying early
            return np.clip(-1.0 + extra + self._term_score_w * float(prev.score),
                        self._clip_lo, self._clip_hi)

        r = 0.0

        # Base per-step cost to discourage wandering/idling
        r += self._step_cost

        # Potential-based distance shaping (policy invariant)
        # Φ(s) = -norm_dist(s), norm_dist ∈ [0,1]
        H, W = cur.grid_h, cur.grid_w
        Dmax = (H - 1) + (W - 1)
        if self._prev_food_dist is not None and Dmax > 0:
            d_prev = float(self._prev_food_dist) / Dmax
            d_cur  = float(manhattan(cur.snake[0], cur.food)) / Dmax
            # r_shape = k * (γ * Φ(s') - Φ(s)) = k * (d_prev - γ*d_cur)
            r += self._k_potential * (self.rules.cfg.gamma * (-d_cur) - (-d_prev))
            # algebraically same as: self._k_potential * (d_prev - γ*d_cur)

        # Bite reward with streak bonus
        if cur.score > prev.score:
            r += self._bite_base + self._bite_streak * float(self._streak)
            # small immediate post-bite momentum bonus (handled by steps_since_bite==0 below)

        # Anti-looping: penalize tiny orbits
        if len(self._recent_heads) >= self._recent_heads.maxlen:
            if len(set(self._recent_heads)) <= 3:
                r += self._loop_penalty

        # Post-bite burst encouragement (brief, decays)
        s = self._steps_since_bite
        if s <= self._post_bite_bonus_steps:
            if len(set(self._recent_heads)) >= 5:           # avoid rewarding tight loops
                r += self._post_bite_step_bonus * float(self._post_bite_bonus_steps - s)

        # Soft starvation ramp after half the threshold (optional)
        if self._steps_since_bite > (self._T_starve // 2):
            ramp = min(0.10, 0.005 * float(self._steps_since_bite - self._T_starve // 2))
            r -= ramp

        return float(np.clip(r, self._clip_lo, self._clip_hi))

