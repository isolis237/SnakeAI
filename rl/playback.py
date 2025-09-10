# rl/playback.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Protocol, Tuple
import queue, threading, time, random
from core.interfaces import Snapshot, SnapshotSink

@dataclass
class Episode:
    frames: List[Snapshot]
    overlay: str = ""
    gap_sec: float = 0.35  # pause between episodes

class EpisodeRecorder:
    """
    Lightweight per-episode recorder: start -> capture(snapshot) -> finish(summary) -> Episode
    """
    def __init__(self, stream_prob: float = 1.0, rng_seed: Optional[int] = None):
        self._rng = random.Random(rng_seed)
        self._stream_prob = max(0.0, min(1.0, stream_prob))
        self._recording = False
        self._frames: List[Snapshot] = []
        self._ep_idx: Optional[int] = None

    def start(self, ep_index: int) -> None:
        self._recording = (self._rng.random() < self._stream_prob)
        self._frames.clear()
        self._ep_idx = ep_index

    def is_recording(self) -> bool:
        return self._recording

    def capture(self, snap: Snapshot) -> None:
        if self._recording:
            self._frames.append(snap)

    def _overlay_from_summary(self, summary: Dict[str, Any]) -> str:
        ep = self._ep_idx if self._ep_idx is not None else "-"
        r  = float(summary.get("reward", 0.0))
        ln = int(summary.get("steps", 0))
        sc = int(summary.get("final_score", 0))
        dr = str(summary.get("death_reason", ""))
        return f"Episode {ep}  |  R={r:.2f}  len={ln}  score={sc}  death={dr}"

    def finish(self, summary: Dict[str, Any]) -> Optional[Episode]:
        if not self._recording or not self._frames:
            self._recording = False
            self._frames.clear()
            self._ep_idx = None
            return None

        ep = Episode(frames=list(self._frames), overlay=self._overlay_from_summary(summary))
        self._recording = False
        self._frames.clear()
        self._ep_idx = None
        return ep

class EpisodeQueuePlayer:
    """
    Single background thread:
      - consumes whole Episode objects from a Queue
      - renders at a steady FPS via the provided sink
    """
    def __init__(self, sink: SnapshotSink, fps: int = 10, max_queue: int = 8):
        self._sink = sink
        self._fps = max(1, int(fps))
        self._q: "queue.Queue[Episode]" = queue.Queue(maxsize=max_queue)
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False
        self._grid: Optional[Tuple[int, int]] = None

    def start(self, grid_w: int, grid_h: int) -> None:
        if self._started:
            return
        self._grid = (grid_w, grid_h)
        self._sink.start(grid_w, grid_h)
        self._stop.clear()
        self._t = threading.Thread(target=self._run, name="EpisodeQueuePlayer", daemon=True)
        self._t.start()
        self._started = True

    def enqueue(self, ep: Episode) -> None:
        if not self._started:
            raise RuntimeError("EpisodeQueuePlayer.start() must be called first")
        try:
            self._q.put_nowait(ep)
        except queue.Full:
            # Drop the oldest by draining one, then add (keeps UI fresh)
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(ep)
            except queue.Full:
                pass  # if still full, drop silently

    def _run(self) -> None:
        frame_dt = 1.0 / float(self._fps)
        while not self._stop.is_set():
            try:
                ep = self._q.get(timeout=0.25)
            except queue.Empty:
                continue

            if ep.overlay:
                self._sink.set_overlay(ep.overlay)

            for snap in ep.frames:
                t0 = time.perf_counter()
                self._sink.push(snap)
                # simple pacing; if training thread is heavy, this still keeps UI smooth
                elapsed = time.perf_counter() - t0
                time_to_sleep = max(0.0, frame_dt - elapsed)
                time.sleep(time_to_sleep)

            if ep.gap_sec > 0:
                time.sleep(ep.gap_sec)

    def close(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)
        self._sink.close()
        self._t = None
        self._started = False

class LiveHook:
    """
    Tiny adapter for your trainer:
      - call on_episode_start(ep)
      - call maybe_render(step_stats, step_info={ "snapshot": Snapshot })
      - call on_episode_end(summary, last_info)
    """
    def __init__(
        self,
        player: EpisodeQueuePlayer,
        recorder: EpisodeRecorder,
        cfg: AppConfig
    ):
        self._player = player
        self._rec = recorder
        self._grid_w, self._grid_h = cfg.grid_w, cfg.grid_h
        self._best_key = cfg.best_metric
        self._best_val = float("-inf")
        self._best_ep: Optional[Episode] = None
        self._started = False

    def _ensure_started(self):
        if not self._started:
            self._player.start(self._grid_w, self._grid_h)
            self._started = True

    def on_episode_start(self, ep_index: int) -> None:
        self._ensure_started()
        self._rec.start(ep_index)

    def maybe_render(self, step_stats: Optional[Dict[str, Any]], step_info: Dict[str, Any]) -> None:
        snap: Optional[Snapshot] = step_info.get("snapshot")
        if snap is not None:
            self._rec.capture(snap)

    def on_episode_end(self, episode_summary: Dict[str, Any], last_info: Dict[str, Any]) -> None:
        ep = self._rec.finish(episode_summary)
        if ep is not None:
            # enqueue immediately for playback
            self._player.enqueue(ep)
            # track "best" for a final encore if you want it later
            val = float(episode_summary.get(self._best_key, episode_summary.get("reward", 0.0)))
            if val > self._best_val:
                self._best_val = val
                self._best_ep = ep

    def replay_best(self) -> None:
        if self._best_ep is not None:
            self._player.enqueue(self._best_ep)

    def close(self) -> None:
        self._player.close()

    # --- NEW: for back-compat with your trainer
    def ensure_started(self) -> None:
        """Back-compat shim; trainer calls this before the loop."""
        self._ensure_started()

    # --- NEW: for trainer’s streaming checks
    def is_streaming(self) -> bool:
        """Return whether the current episode is being recorded."""
        return self._rec.is_recording()
