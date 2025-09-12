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
from rl.playback import EpisodeRecorder, EpisodeQueuePlayer
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
        return 0


class AStarPolicy(Policy):
    """
    Single-step replanning A* to the food.
    - Obstacles = snake body (tail optionally treated as free).
    - Walls are implicit boundaries.
    - If no path: pick safest greedy step (toward food that doesn’t die immediately).
    """
    def __init__(self, consider_tail_free: bool = True):
        self.consider_tail_free = consider_tail_free
        self._snap: Optional[Snapshot] = None
        self._relative_actions: bool = True

    def act(self, obs: np.ndarray) -> int:
        snap = self._snap
        assert snap is not None, "AStarPolicy.act called before set_snapshot()"
        head = snap.snake[0]
        food = snap.food
        H, W = snap.grid_h, snap.grid_w

        blocked = set(snap.snake)
        if self.consider_tail_free and len(snap.snake) > 1:
            blocked.discard(snap.snake[-1])  # tail likely moves

        path = self._astar(head, food, blocked, W, H)

        if path and len(path) >= 2:
            nxt = path[1]
            dx, dy = (nxt[0]-head[0], nxt[1]-head[1])
            next_dir = (int(np.sign(dx)), int(np.sign(dy)))
        else:
            next_dir = self._safe_greedy(head, food, blocked, W, H, cur_dir=snap.dir)

        if self._relative_actions:
            return rel_action_from_dirs(snap.dir, next_dir)
        else:
            return ABS_DIRS.index(next_dir)

    def act_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        return np.array([self.act(obs_batch[0])], dtype=np.int64)

    def set_snapshot(self, snap: Snapshot, relative_actions: bool) -> None:
        self._snap = snap
        self._relative_actions = relative_actions

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
                # reconstruct path
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
                        heapq.heappush(openq, (f, v))
                        in_open.add(v)
        return None

    def _safe_greedy(
        self, head, food, blocked: set, W: int, H: int, cur_dir: Tuple[int,int]
    ) -> Tuple[int,int]:
        moves = []
        for d in ABS_DIRS:
            nx = (head[0] + d[0], head[1] + d[1])
            if 0 <= nx[0] < W and 0 <= nx[1] < H and nx not in blocked:
                moves.append((manhattan(nx, food), d))
        if moves:
            moves.sort(key=lambda t: t[0])
            return moves[0][1]
        nx = (head[0] + cur_dir[0], head[1] + cur_dir[1])
        if 0 <= nx[0] < W and 0 <= nx[1] < H and nx not in blocked:
            return cur_dir
        for d in ABS_DIRS:
            nx = (head[0] + d[0], head[1] + d[1])
            if 0 <= nx[0] < W and 0 <= nx[1] < H and nx not in blocked:
                return d
        return cur_dir  # doomed; move forward
        

def main():
    # --- Config & Env ---
    cfg = AppConfig().with_(relative_actions=False) 
    rules = Rules(cfg)
    env = SnakeEnv(cfg, rules)

    # --- Playback wiring ---
    sink = None
    player: Optional[EpisodeQueuePlayer] = None
    recorder: Optional[EpisodeRecorder] = None
    if getattr(cfg, "live_view", False):
        sink = LiveViewer(cfg)
        player = EpisodeQueuePlayer(sink=sink, fps=cfg.fps, max_queue=cfg.queue_max)
        player.start(cfg.grid_w, cfg.grid_h)   # start before enqueue
        recorder = EpisodeRecorder()

    # --- Logger ---
    logger = CSVLogger("runs/snake_astar/logs.csv", fieldnames=ALL_KEYS)
    global_step_counter = {"x": 0}
    on_episode_end = make_episode_logger(
        logger=logger,
        ema_reward=EMA(0.05),
        ema_length=EMA(0.05),
        win_reward=WindowedStat(100),
        win_length=WindowedStat(100),
        step_getter=lambda: global_step_counter["x"],
    )
    hooks = TrainHooks(on_episode_end=on_episode_end)

    # --- Policy ---
    policy = AStarPolicy(consider_tail_free=True)

    print("=== Snake A* ===")
    print(f"grid: {cfg.grid_w}x{cfg.grid_h}  relative_actions: {cfg.relative_actions}  live_view: {getattr(cfg, 'live_view', False)}")

    try:
        high_score = 0
        # --- Run episodes (no learning) ---
        for ep in range(cfg.episodes):
            obs = env.reset(seed=cfg.seed)
            total_r = 0.0
            step = 0
            terminated = False

            # start recording this episode
            if recorder is not None:
                recorder.start(ep)
                recorder.capture(env.get_snapshot())

            max_steps = cfg.max_ep_steps if cfg.max_ep_steps is not None else 10**9

            while not terminated and step < max_steps:
                snap = env.get_snapshot()

                if recorder is not None:
                    recorder.capture(snap)

                policy.set_snapshot(snap, relative_actions=cfg.relative_actions)
                a = policy.act(obs)
                res = env.step(a)
                obs = res.obs
                total_r += res.reward
                terminated = res.terminated or res.truncated
                step += 1

            # final frame + enqueue episode for playback
            if recorder is not None and player is not None:
                recorder.capture(env.get_snapshot())
                last = env.get_snapshot()
                summary = {
                    "reward": total_r,
                    "steps": step,
                    "final_score": last.score,
                    "death_reason": last.reason,
                }
                episode_obj = recorder.finish(summary)
                player.enqueue(episode_obj)
                high_score = max(high_score, last.score)

            # logging
            global_step_counter["x"] += step
            if hooks.on_episode_end:
                hooks.on_episode_end(ep, {
                    "reward": total_r,
                    "steps": step,
                    "final_score": env.get_snapshot().score,
                    "death_reason": env.get_snapshot().reason,
                })

        # ---- graceful playback shutdown ----
        if player is not None:
            # let the queue drain (so last episode finishes)
            player.wait_idle(timeout=10.0)   # wait up to 10s; returns early if idle
            player.close()                   # closes thread and sink

        print(f"Highest score: {high_score}")

    except KeyboardInterrupt:
        # allow clean exit on Ctrl-C
        if player is not None:
            player.close()
        raise
    finally:
        logger.close()
