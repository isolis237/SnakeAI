# rl/async_episode_player.py
from dataclasses import dataclass
from typing import List, Optional, Protocol
import queue, threading, time
from core.interfaces import Snapshot, SnapshotSink

@dataclass
class _WorkItem:
    snaps: List[Snapshot]
    gap_sec: float
    overlay: Optional[str] = None   # ← NEW

class AsyncEpisodePlayer:
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

    def enqueue_episode(self, frames: List[Snapshot], gap_sec: float = 0.35, overlay: Optional[str] = None) -> None:
        if not frames:
            return
        item = _WorkItem(frames, gap_sec, overlay)
        try:
            self._q.put(item, block=False)
        except queue.Full:
            # Drop oldest to stay responsive
            try: self._q.get_nowait()
            except queue.Empty: pass
            self._q.put_nowait(item)

    def _run(self) -> None:
        frame_dt = 1.0 / self._fps
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            # ← NEW: set episode overlay once before playing its frames
            if item.overlay and hasattr(self._sink, "set_overlay"):
                try:
                    self._sink.set_overlay(item.overlay)
                except Exception:
                    pass

            for snap in item.snaps:
                if self._stop.is_set(): break
                t0 = time.perf_counter()
                self._sink.push(snap)
                dt = time.perf_counter() - t0
                if frame_dt > dt:
                    time.sleep(frame_dt - dt)

            if item.gap_sec and not self._stop.is_set():
                time.sleep(item.gap_sec)

        try: self._sink.close()
        except Exception: pass

    def qsize(self) -> int:
        return self._q.qsize()

    def capacity(self) -> int:
        return self._q.maxsize

    def close(self) -> None:
        if not self._started: return
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=3.0)
        self._started = False
