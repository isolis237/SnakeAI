# snake/viz/live_viewer.py
from __future__ import annotations
import multiprocessing as mp
from typing import Optional
from core.interfaces import Snapshot
from viz.snapshot_sink import SnapshotSink
from viz.renderer_pygame import PygameRenderer
from viz.render_iface import RenderConfig

def _viewer_proc(q: mp.Queue, grid_w: int, grid_h: int, fps: int, record_dir: Optional[str], title: str, cell_px: int, grid_lines: bool, show_hud: bool):
    # Everything Pygame-related stays in this child process.
    ren = PygameRenderer()
    ren.open(grid_w, grid_h, RenderConfig(
        cell_px=cell_px,
        title=title,
        grid_lines=grid_lines,
        show_hud=show_hud,
        record_dir=record_dir,
    ))
    # Main viewer loop
    try:
        while True:
            item = q.get()
            if item is None:  # sentinel to exit
                break
            ren.draw(item)    # item is a Snapshot
            ren.tick(fps)
    except KeyboardInterrupt:
        pass
    finally:
        ren.close()

class LiveViewer(SnapshotSink):
    """
    Push snapshots to a separate process that displays (and optionally records) frames.
    Non-blocking for the trainer (Queue puts are bounded & try/except).
    """
    def __init__(self, fps: int = 10, record_dir: Optional[str] = None, title: str = "Snake — Training", cell_px: int = 24, grid_lines: bool = False, show_hud: bool = True, queue_max: int = 4):
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
        ctx = mp.get_context("spawn")  # safer across platforms
        self._q = ctx.Queue(maxsize=self._queue_max)
        self._proc = ctx.Process(
            target=_viewer_proc,
            args=(self._q, grid_w, grid_h, self._fps, self._record_dir, self._title, self._cell_px, self._grid_lines, self._show_hud),
            daemon=True,
        )
        self._proc.start()

    def push(self, snap: Snapshot) -> None:
        if not self._q:
            return
        # Non-blocking push so training never stalls; drop if viewer is behind.
        try:
            self._q.put_nowait(snap)
        except Exception:
            pass  # queue full → drop frame

    def close(self) -> None:
        if self._q:
            try:
                self._q.put(None)  # sentinel
            except Exception:
                pass
        if self._proc:
            self._proc.join(timeout=2.0)
        self._proc = None
        self._q = None
