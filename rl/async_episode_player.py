# rl/async_episode_player.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Protocol
import queue, threading, time

# Your Snapshot type; import from your project if needed
from core.interfaces import Snapshot

class SnapshotSink(Protocol):
    def start(self, grid_w: int, grid_h: int) -> None: ...
    def push(self, snap: Snapshot) -> None: ...
    def close(self) -> None: ...

@dataclass
class _WorkItem:
    snaps: List[Snapshot]           # whole episode
    gap_sec: float                  # pause after episode

class AsyncEpisodePlayer:
    """
    Replays *complete* episodes at a stable FPS on a background thread.
    Training enqueues completed episodes. This thread is the only one
    touching the sink (prevents pygame/video thread-safety issues).
    """
    def __init__(self, sink: SnapshotSink, fps: float = 15.0, max_episodes_buffered: int = 8):
        self._sink = sink
        self._fps = max(1e-3, float(fps))
        self._q: queue.Queue[_WorkItem] = queue.Queue(maxsize=max_episodes_buffered)
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False

    def start(self, grid_w: int, grid_h: int) -> None:
        if self._started: return
        self._sink.start(grid_w, grid_h)
        self._stop.clear()
        self._t = threading.Thread(target=self._run, name="AsyncEpisodePlayer", daemon=True)
        self._t.start()
        self._started = True

    def enqueue_episode(self, frames: List[Snapshot], gap_sec: float = 0.35) -> None:
        """
        Non-blocking: if buffer is full, drop oldest to stay responsive.
        """
        if not frames:
            return
        try:
            self._q.put(_WorkItem(frames, gap_sec), block=False)
        except queue.Full:
            # drop the oldest to always accept the newest
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(_WorkItem(frames, gap_sec))

    def _run(self) -> None:
        frame_dt = 1.0 / self._fps
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Play the episode at steady FPS
            for snap in item.snaps:
                if self._stop.is_set():
                    break
                t0 = time.perf_counter()
                self._sink.push(snap)
                # simple pacing (don’t overspec colors/styles; let sink paint)
                elapsed = time.perf_counter() - t0
                sleep_for = max(0.0, frame_dt - elapsed)
                if sleep_for:
                    time.sleep(sleep_for)

            # brief gap between episodes (avoid “rushes”)
            if item.gap_sec and not self._stop.is_set():
                time.sleep(item.gap_sec)

        # graceful end
        try:
            self._sink.close()
        except Exception:
            pass

    def close(self) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=3.0)
        self._started = False
