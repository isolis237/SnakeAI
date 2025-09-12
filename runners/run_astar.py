# runners/run_astar.py
from __future__ import annotations

from typing import Tuple, List, Optional, Dict
import heapq
import numpy as np

from config import AppConfig
from core.snake_rules import Rules
from core.snake_env import SnakeEnv, manhattan
from core.interfaces import Policy, Snapshot
from rl.logging import CSVLogger, make_episode_logger, ALL_KEYS
from rl.metrics import EMA, WindowedStat
from rl.trainer import TrainHooks
from rl.playback import EpisodeRecorder, EpisodeQueuePlayer, LiveHook
from viz.live_viewer import LiveViewer


# -------- A* policy helpers --------

ABS_DIRS: List[Tuple[int, int]] = [(1,0), (0,1), (-1,0), (0,-1)]  # R, D, L, U

def rel_action_from_dirs(cur_dir: Tuple[int,int], next_dir: Tuple[int,int]) -> int:
    """
    Relative action mapping for 3-action scheme:
      0 = turn left, 1 = straight, 2 = turn right
    Assumes cur_dir and next_dir are in ABS_DIRS.
    """
    cur_i = ABS_DIRS.index(cur_dir)
    nxt_i = ABS_DIRS.index(next_dir)
    delta = (nxt_i - cur_i) % 4
    if delta == 0:   # same direction
        return 1
    elif delta == 1: # right
        return 2
    elif delta == 3: # left
        return 0
    else:
        # 180° reversal (illegal for len>1 in absolute mode; relative mode shouldn’t ask for it)
        # choose left by default to avoid crashing
        return 0


class AStarPolicy(Policy):
    """
    Single-step replanning A* to the food:
      - Obstacles = snake body (optionally treat tail as free since it moves)
      - Walls are implicit boundaries
      - If no path: pick safest greedy step (toward food that doesn’t die immediately)
    """
    def __init__(self, consider_tail_free: bool = True):
        self.consider_tail_free = consider_tail_free

    def act(self, obs: np.ndarray) -> int:
        # We don’t rely on obs; caller provides a snapshot via set_snapshot before act()
        snap = self._snap
        H, W = snap.grid_h, snap.grid_w
        head = snap.snake[0]
        food = snap.food

        blocked = set(snap.snake)  # all body segments including head
        if self.consider_tail_free and len(snap.snake) > 1:
            # Tail will move unless we eat this step; free it to reduce false dead ends
            tail = snap.snake[-1]
            blocked.discard(tail)

        # Plan shortest path on grid using A*
        path = self._astar(head, food, blocked, W, H)

        # Decide next absolute direction
        if path and len(path) >= 2:
            nxt = path[1]
            dx, dy = (nxt[0]-head[0], nxt[1]-head[1])
            next_dir = (int(np.sign(dx)), int(np.sign(dy)))
        else:
            # No path: try safe greedy move
            next_dir = self._safe_greedy(head, food, blocked, W, H, cur_dir=snap.dir)

        # Map to action depending on control mode
        if snap:  # relative vs absolute
            if self._relative_actions:
                return rel_action_from_dirs(snap.dir, next_dir)
            else:
                return ABS_DIRS.index(next_dir)

    def act_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        # Not used here; single-env eval
        return np.array([self.act(obs_batch[0])])

    # ----- plumbing to feed snapshot & mode -----
    def set_snapshot(self, snap: Snapshot, relative_actions: bool) -> None:
        self._snap = snap
        self._relative_actions = relative_actions

    # ----- A* implementation -----
    def _astar(self, start, goal, blocked: set, W: int, H: int) -> Optional[List[Tuple[int,int]]]:
        def inb(p): return 0 <= p[0] < W and 0 <= p[1] < H
        def neigh(p):
            for dx, dy in ABS_DIRS:
                q = (p[0]+dx, p[1]+dy)
                if inb(q) and q not in blocked:
                    yield q

        openq: List[Tuple[float, Tuple[int,int]]] = []
        g: Dict[Tuple[int,int], float] = {start: 0.0}
        parent: Dict[Tuple[int,int], Tuple[int,int]] = {}
        f0 = manhattan(start, goal)
        heapq.heappush(openq, (f0, start))
        in_open = {start}

        while openq:
            _, u = heapq.heappop(openq); in_open.discard(u)
            if u == goal:
                # reconstruct
                path = [u]
                while u in parent:
                    u = parent[u]
                    path.append(u)
                return list(reversed(path))
            for v in neigh(u):
                cand = g[u] + 1.0
                if cand < g.get(v, 1e9):
                    g[v] = cand
                    parent[v] = u
                    f = cand + manhattan(v, goal)
                    if v not in in_open:
                        heapq.heappush(openq, (f, v)); in_open.add(v)
        return None

    def _safe_greedy(
        self, head, food, blocked: set, W: int, H: int, cur_dir: Tuple[int,int]
    ) -> Tuple[int,int]:
        # Rank moves by (closer to food first), but only those that keep us alive
        moves = []
        for d in ABS_DIRS:
            nx = (head[0] + d[0], head[1] + d[1])
            if 0 <= nx[0] < W and 0 <= nx[1] < H and nx not in blocked:
                moves.append((manhattan(nx, food), d))
        if moves:
            moves.sort(key=lambda t: t[0])
            return moves[0][1]
        # If totally trapped, at least try to avoid 180° reverse in absolute mode
        # Fall back to current direction if possible
        nx = (head[0] + cur_dir[0], head[1] + cur_dir[1])
        if 0 <= nx[0] < W and 0 <= nx[1] < H and nx not in blocked:
            return cur_dir
        # Otherwise just pick any legal direction
        for d in ABS_DIRS:
            nx = (head[0] + d[0], head[1] + d[1])
            if 0 <= nx[0] < W and 0 <= nx[1] < H and nx not in blocked:
                return d
        return cur_dir  # doomed; move forward
        

def main():
    # --- Config & Env ---
    cfg = AppConfig().with_(relative_actions=True)
    rules = Rules(cfg)
    env = SnakeEnv(cfg, rules)

    # --- Live view (optional) ---
    live_hook = None
    if cfg.live_view:
        sink = LiveViewer(cfg)
        player = EpisodeQueuePlayer(sink=sink, fps=cfg.fps, max_queue=cfg.queue_max)
        live_hook = LiveHook(player=player, recorder=EpisodeRecorder(), cfg=cfg)

    # --- Logger (reuse your CSV format for apples/len, etc.) ---
    logger = CSVLogger("runs/snake_astar/logs.csv", fieldnames=ALL_KEYS)
    on_episode_end = make_episode_logger(
        logger=logger, ema_reward=EMA(0.05), ema_length=EMA(0.05),
        win_reward=WindowedStat(100), win_length=WindowedStat(100),
        step_getter=lambda: 0  # no learner step counter for A*
    )
    hooks = TrainHooks(on_episode_end=on_episode_end)

    # --- Policy ---
    policy = AStarPolicy(consider_tail_free=True)

    print("=== Snake A* ===")
    print(f"grid: {cfg.grid_w}x{cfg.grid_h}  relative_actions: {cfg.relative_actions}")

    # --- Evaluate for cfg.episodes episodes (no learning) ---
    for ep in range(cfg.episodes):
        obs = env.reset(seed=cfg.seed)
        total_r = 0.0
        step = 0
        terminated = False
        while not terminated and (cfg.max_ep_steps or 10**9) > step:
            snap = env.get_snapshot()
            policy.set_snapshot(snap, relative_actions=cfg.relative_actions)
            a = policy.act(obs)
            res = env.step(a)
            obs = res.obs
            total_r += res.reward
            terminated = res.terminated or res.truncated
            step += 1

        # Push episode summary to logger
        # make_episode_logger expects info fields in trainer; we mimic minimal keys
        info = {
            "epis/reward": total_r,
            "epis/length": step,
            "epis/score": env.get_snapshot().score,
        }
        if hooks.on_episode_end:
            hooks.on_episode_end(ep, info)

        # Optional: stream the full episode via live_hook (if enabled)
        if live_hook is not None and np.random.rand() < getattr(cfg, "stream_ep_prob", 0.0):
            # No recorder integration shown here; LiveHook in your project may already capture episodes
            pass

    logger.close()


if __name__ == "__main__":
    main()
