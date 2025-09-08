# snake/viz/live_viewer.py
from __future__ import annotations
import multiprocessing as mp
from typing import Optional, Tuple, Any
from interfaces import Snapshot
from config import RenderConfig
from viz.renderer_pygame import PygameRenderer
from viz.snapshot_sink import SnapshotSink

Msg = Tuple[str, Any]  # ("frame", Snapshot) or ("overlay", str) or ("quit", None)

def _viewer_proc(q: mp.Queue, grid_w: int, grid_h: int, fps: int,
                 record_dir: Optional[str], title: str, cell_px: int,
                 grid_lines: bool, show_hud: bool):
    ren = PygameRenderer()
    ren.open(grid_w, grid_h, RenderConfig(
        cell_px=cell_px,
        title=title,
        grid_lines=grid_lines,
        show_hud=show_hud,
        record_dir=record_dir,
    ))
    try:
        while True:
            msg: Msg = q.get()
            if msg is None:
                break
            kind, payload = msg
            if kind == "quit":
                break
            elif kind == "overlay":
                ren.set_overlay(payload)  # payload is str
            elif kind == "frame":
                snap: Snapshot = payload
                ren.draw(snap)
                ren.tick(fps)
    except KeyboardInterrupt:
        pass
    finally:
        ren.close()

class LiveViewer(SnapshotSink):
    def __init__(self, fps: int = 10, record_dir: Optional[str] = None,
                 title: str = "Snake — Training", cell_px: int = 24,
                 grid_lines: bool = False, show_hud: bool = True,
                 queue_max: int = 4):
        self._fps = fps
        self._record_dir = record_dir
        self._title = title
        self._cell_px = cell_px
        self._grid_lines = grid_lines
        self._show_hud = show_hud
        self._q: Optional[mp.Queue] = None
        self._proc: Optional[mp.Process] = None
        self._queue_max = queue_max

    def start(self, grid_w: int, grid_h: int) -> None:
        if self._proc is not None:
            return
        ctx = mp.get_context("spawn")
        self._q = ctx.Queue(maxsize=self._queue_max)
        self._proc = ctx.Process(
            target=_viewer_proc,
            args=(self._q, grid_w, grid_h, self._fps,
                  self._record_dir, self._title, self._cell_px,
                  self._grid_lines, self._show_hud),
            daemon=True,
        )
        self._proc.start()

    def set_overlay(self, text: str) -> None:
        # Send a control message to the child process
        if self._q:
            try:
                self._q.put_nowait(("overlay", text))
            except Exception:
                pass

    def push(self, snap: Snapshot) -> None:
        if not self._q:
            return
        try:
            self._q.put_nowait(("frame", snap))
        except Exception:
            pass  # drop frame if queue full

    def close(self) -> None:
        if self._q:
            try:
                self._q.put(("quit", None))
            except Exception:
                pass
        if self._proc:
            self._proc.join(timeout=2.0)
        self._proc = None
        self._q = None
