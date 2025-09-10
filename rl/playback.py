# rl/playback.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import queue, threading, time, random
from collections import deque
from core.interfaces import Snapshot, SnapshotSink
from config import AppConfig  # for cfg fields in LiveHook

@dataclass
class Episode:
    frames: List[Snapshot]
    overlay: str = ""
    gap_sec: float = 0.35  # pause between episodes

class EpisodeRecorder:
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
    def __init__(self, sink: SnapshotSink, fps: int = 10, max_queue: int = 8):
        self._sink = sink
        self._fps = max(1, int(fps))
        self._q: "queue.Queue[Episode]" = queue.Queue(maxsize=max_queue)
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False
        self._playing = threading.Event()

    def start(self, grid_w: int, grid_h: int) -> None:
        if self._started:
            return
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
            # drop oldest then try again
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(ep)
            except queue.Full:
                pass

    def qsize(self) -> int:
        return self._q.qsize()

    def drain_queue(self) -> None:               
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        frame_dt = 1.0 / float(self._fps)
        while not self._stop.is_set():
            try:
                ep = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            self._playing.set()                    
            if ep.overlay:
                self._sink.set_overlay(ep.overlay)

            for snap in ep.frames:
                t0 = time.perf_counter()
                self._sink.push(snap)
                time.sleep(max(0.0, frame_dt - (time.perf_counter() - t0)))

            self._playing.clear()                  

            if ep.gap_sec > 0:
                time.sleep(ep.gap_sec)

    def play_now(self, ep: Episode, clear_queue: bool = True) -> None:  # NEW
        if not self._started:
            raise RuntimeError("start() must be called first")
        if clear_queue:
            self.drain_queue()
        self.enqueue(ep)  # there will be room now

    def wait_idle(self, timeout: Optional[float] = None) -> bool:       # NEW
        t0 = time.perf_counter()
        while True:
            if self.qsize() == 0 and not self._playing.is_set():
                return True
            if timeout is not None and (time.perf_counter() - t0) > timeout:
                return False
            time.sleep(0.01)

    def close(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)
        self._sink.close()
        self._t = None
        self._started = False

class LiveHook:
    """
    Trainer-facing hook. Keeps a recent-episodes cache and auto-fills the player queue
    to avoid multi-second gaps when training produces episodes slower than playback.
    """
    def __init__(
        self,
        player: EpisodeQueuePlayer,
        recorder: EpisodeRecorder,
        cfg: AppConfig,
        *,
        recent_cap: int = 8,
        min_fill: int = 2,
        target_fill: int = 4,
        downsample_stride: int = 1,
        feeder_interval_sec: float = 0.2,
        rng_seed: Optional[int] = None,
        best_selector: Optional[Callable[[Dict[str, Any]], float]] = None,
    ):
        self._best_selector = best_selector or (lambda s: float(
            s.get("final_score", s.get("reward", 0.0))
        ))
        self._player = player
        self._rec = recorder
        self._grid_w, self._grid_h = cfg.grid_w, cfg.grid_h
        self._best_key = cfg.best_metric
        self._best_val = float("-inf")
        self._best_ep: Optional[Episode] = None

        self._recent: deque[List[Snapshot]] = deque(maxlen=recent_cap)
        self._recent_overlays: deque[str] = deque(maxlen=recent_cap)
        self._stride = max(1, int(downsample_stride))
        self._min_fill = int(min_fill)
        self._target_fill = int(target_fill)
        self._feeder_interval = max(0.05, float(feeder_interval_sec))
        self._rng = random.Random(rng_seed)

        self._started = False
        self._feeder_stop = threading.Event()
        self._feeder_t: Optional[threading.Thread] = None

    # --- lifecycle
    def ensure_started(self) -> None:
        if not self._started:
            self._player.start(self._grid_w, self._grid_h)
            # start feeder
            self._feeder_stop.clear()
            self._feeder_t = threading.Thread(target=self._feeder_run, name="EpisodeFeeder", daemon=True)
            self._feeder_t.start()
            self._started = True

    def _feeder_run(self) -> None:
        while not self._feeder_stop.is_set():
            try:
                cur = self._player.qsize()
                if cur < self._min_fill and self._recent:
                    need = max(0, self._target_fill - cur)
                    for _ in range(need):
                        k = self._rng.randrange(len(self._recent))
                        frames = list(self._recent[k])  # copy
                        overlay = self._recent_overlays[k]
                        self._player.enqueue(Episode(frames=frames, overlay=overlay, gap_sec=0.25))
                time.sleep(self._feeder_interval)
            except Exception:
                time.sleep(self._feeder_interval)

    # --- trainer API
    def start_episode(self, ep_index: int) -> None:
        self.ensure_started()
        self._rec.start(ep_index)

    def record_snapshot(self, snap: Snapshot) -> None:
        self._rec.capture(snap)

    def end_episode(self, episode_summary: Dict[str, Any], last_info: Dict[str, Any]) -> None:
        ep = self._rec.finish(episode_summary)
        if ep is None:
            return

        self._player.enqueue(ep)

        val = self._best_selector(episode_summary)   # NEW
        if val > self._best_val:
            self._best_val = val
            # keep an independent copy so later replays can’t be affected
            self._best_ep = Episode(frames=list(ep.frames), overlay=ep.overlay, gap_sec=0.0)

        # cache a downsampled copy for feeder
        if ep.frames:
            cached = ep.frames[::self._stride] if self._stride > 1 else ep.frames
            if cached:
                self._recent.append(list(cached))
                self._recent_overlays.append(ep.overlay)

    def replay_best_and_wait(self, timeout: Optional[float] = 15.0) -> None:
        if not self._best_ep:
            return
        # If you run a feeder, stop it first so it won’t insert more episodes.
        if hasattr(self, "_feeder_stop"):
            self._feeder_stop.set()
            if getattr(self, "_feeder_t", None):
                self._feeder_t.join(timeout=0.5)

        # Play best immediately, no gaps, at the front of the line.
        best = Episode(frames=list(self._best_ep.frames),
                       overlay=f"BEST — score/reward={self._best_val:.2f}",
                       gap_sec=0.0)
        self._player.play_now(best, clear_queue=True)
        self._player.wait_idle(timeout=timeout)

    def replay_best(self) -> None:
        # legacy non-blocking version (keep if trainer still calls it)
        if self._best_ep is not None:
            self._player.play_now(self._best_ep, clear_queue=False)

    def close(self) -> None:
        self._feeder_stop.set()
        if self._feeder_t:
            self._feeder_t.join(timeout=0.5)
        self._player.close()

    # (optional) legacy compatibility
    def is_streaming(self) -> bool:
        return self._rec.is_recording()
