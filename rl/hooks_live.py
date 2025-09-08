# rl/hooks_live.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import random, threading, time
from collections import deque
from interfaces import Snapshot
from rl.async_episode_player import AsyncEpisodePlayer  # NEW

class LiveRenderHook:
    """
    Episode-level streaming:
      - With probability stream_episode_prob, we RECORD every step of THIS episode.
      - On episode end, we enqueue the WHOLE episode to the AsyncEpisodePlayer,
        which replays at a steady FPS independent from training speed.
      - We still track the best episode for a final replay if you want that too.
    """
    def __init__(
        self,
        sink_player: AsyncEpisodePlayer,      # <- use the player, not raw sink
        grid_w: int,
        grid_h: int,
        stream_episode_prob: float = 1.0,
        best_metric: str = "reward",
        rng_seed: Optional[int] = None,
        enqueue_best_for_final_replay: bool = True,
        recent_cap: int = 8,            # how many full episodes to keep in cache
        min_fill: int = 2,              # keep at least this many enqueued
        target_fill: int = 4,           # try to fill up to this many
        downsample_stride: int = 1,     # store every Nth frame in cache; 1 = keep all
        feeder_interval_sec: float = 0.2,
    ):
        self.player = sink_player
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.stream_episode_prob = max(0.0, min(1.0, stream_episode_prob))
        self.best_metric = best_metric
        self._started = False
        self._rng = random.Random(rng_seed)

        self._stream_this_episode = False
        self._curr_episode_frames: List[Snapshot] = []
        self._curr_ep_idx: Optional[int] = None  

        self._best_value: float = float("-inf")
        self._best_episode_frames: List[Snapshot] = []
        self._best_ep_idx: Optional[int] = None  
        self._keep_best = enqueue_best_for_final_replay

        self._recent: deque[Tuple[List[Snapshot], str]] = deque(maxlen=recent_cap)
        self._min_fill = int(min_fill)
        self._target_fill = int(target_fill)
        self._stride = max(1, int(downsample_stride))
        self._feeder_interval = max(0.05, float(feeder_interval_sec))
        self._feeder_stop = threading.Event()
        self._feeder_t: Optional[threading.Thread] = None

    # --- lifecycle ---
    def ensure_started(self) -> None:
        if not self._started:
            self.player.start(self.grid_w, self.grid_h)
            # start feeder
            self._feeder_stop.clear()
            self._feeder_t = threading.Thread(target=self._feeder_run,
                                              name="EpisodeFeeder", daemon=True)
            self._feeder_t.start()
            self._started = True

    def _feeder_run(self) -> None:
        # Background: keep the player queue between min_fill and target_fill by sampling recent episodes
        while not self._feeder_stop.is_set():
            try:
                cur = self.player.qsize()
                if cur < self._min_fill and self._recent:
                    need = max(0, self._target_fill - cur)
                    for _ in range(need):
                        # pick a random cached episode
                        frames, overlay = self._rng.choice(list(self._recent))
                        # Enqueue a COPY to avoid accidental mutation
                        self.player.enqueue_episode(list(frames), gap_sec=0.35, overlay=overlay)
                time.sleep(self._feeder_interval)
            except Exception:
                # Keep the feeder resilient; never crash the training loop
                time.sleep(self._feeder_interval)

    def on_episode_start(self, ep_index: int) -> None:
        if not self._started:
            self.ensure_started()
        self._stream_this_episode = (self._rng.random() < self.stream_episode_prob)
        self._curr_episode_frames.clear()
        self._curr_ep_idx = ep_index

    def is_streaming(self) -> bool:
        # “streaming” here means “recording this episode”
        return self._stream_this_episode

    def maybe_render(self, step_stats: Optional[Dict[str, Any]], step_info: Dict[str, Any]) -> None:
        # No-op; we don’t push to sink per-step anymore.
        # We only collect frames; pushing happens in the player thread after ep ends.
        snap: Optional[Snapshot] = step_info.get("snapshot")  # ok if trainer packs it
        if self._stream_this_episode and snap is not None:
            self._curr_episode_frames.append(snap)

    def _make_overlay(self, episode_summary: Dict[str, Any]) -> str:
        ep = self._curr_ep_idx if self._curr_ep_idx is not None else "-"
        r  = float(episode_summary.get("reward", 0.0))
        ln = int(episode_summary.get("steps", 0))
        sc = int(episode_summary.get("final_score", 0))
        dr = str(episode_summary.get("death_reason", ""))
        return f"Episode {ep}  |  R={r:.2f}  len={ln}  score={sc}  death={dr}"

    def collect_snapshot(self, snap: Snapshot) -> None:
        # Helper if trainer calls env.get_snapshot() explicitly per step
        if self._stream_this_episode:
            self._curr_episode_frames.append(snap)

    def _downsample(self, frames: List[Snapshot]) -> List[Snapshot]:
        if self._stride == 1 or len(frames) <= 1:
            return frames
        return frames[::self._stride]

    def on_episode_end(self, episode_summary: Dict[str, Any], last_info: Dict[str, Any]) -> None:
        if not self._started:
            return

        overlay = self._make_overlay(episode_summary)

        # If this episode was selected, enqueue it immediately (as before)
        if self._stream_this_episode and self._curr_episode_frames:
            frames_to_enqueue = list(self._curr_episode_frames)
            self.player.enqueue_episode(frames_to_enqueue, gap_sec=0.35, overlay=overlay)

            # Track best (optional)
            val = (int(episode_summary.get("final_score", 0))
                   if self.best_metric == "final_score"
                   else float(episode_summary.get("reward", 0.0)))
            if self._keep_best and val > self._best_value:
                self._best_value = val
                self._best_episode_frames = list(self._curr_episode_frames)
                self._best_ep_idx = self._curr_ep_idx

        # ALWAYS cache a downsampled copy of the episode to the recent reservoir
        # (If you didn't record frames live, we can fall back to not caching—OR
        #  you can decide to always record *lightly* by collecting frames even when not streaming.)
        if self._curr_episode_frames:
            cached = self._downsample(self._curr_episode_frames)
            if cached:
                self._recent.append((list(cached), overlay))

        # reset per-episode state
        self._curr_episode_frames.clear()
        self._stream_this_episode = False
        self._curr_ep_idx = None

    def replay_best(self) -> None:
        # Instead of pushing directly, enqueue best for one more playback.
        if not (self._started and self._best_episode_frames):
            return
        ep = self._best_ep_idx if self._best_ep_idx is not None else "-"
        overlay = f"BEST EPISODE (ep {ep})  |  {self.best_metric}={self._best_value:.2f}"
        self.player.enqueue_episode(list(self._best_episode_frames), gap_sec=0.0, overlay=overlay)

    def close(self) -> None:
        self.player.close()
