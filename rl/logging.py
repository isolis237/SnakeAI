from __future__ import annotations
import csv, os, time
from typing import Dict, Any, Protocol, Optional

ALL_KEYS = [
    "step",
    # train
    "train/loss", "train/epsilon", "train/replay_size",
    "train/q_mean", "train/q_max", "train/q_min", "train/updates",
    # episode
    "epis/reward", "epis/reward_ema", "epis/reward_mean100",
    "epis/len", "epis/len_ema", "epis/len_mean100",
    "epis/final_score", "epis/death_wall", "epis/death_self", "epis/death_starvation",
]

class Logger(Protocol):
    def log(self, step: int, scalars: Dict[str, Any]) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class CSVLogger:
    """Append-only CSV logger with header auto-discovery or predefined schema."""
    def __init__(self, path: str, fieldnames: list[str] | None = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._fieldnames = fieldnames
        self._file = open(path, "a", newline="")
        self._writer = None

    def log(self, step: int, scalars: Dict[str, Any]) -> None:
        scalars = {"step": step, **scalars}
        if self._writer is None:
            if self._fieldnames is None:
                self._fieldnames = list(scalars.keys())
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self._fieldnames,
                extrasaction="ignore",   # <-- don't crash on unseen keys
            )
            if self._file.tell() == 0:
                self._writer.writeheader()
        self._writer.writerow(scalars)

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass

class TBLogger:
    """Tiny TensorBoard logger (optional dependency)."""
    def __init__(self, logdir: str):
        os.makedirs(logdir, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except Exception as e:
            raise RuntimeError("TensorBoard is not available") from e
        self.w = SummaryWriter(log_dir=logdir)
        self._start = time.time()

    def log(self, step: int, scalars: Dict[str, Any]) -> None:
        for k, v in scalars.items():
            # Only scalar-friendly values
            if isinstance(v, (int, float)):
                self.w.add_scalar(k, v, step)

    def flush(self) -> None:
        self.w.flush()

    def close(self) -> None:
        self.w.flush(); self.w.close()

class MultiLogger:
    """Fan-out to multiple loggers."""
    def __init__(self, *loggers: Logger):
        self._loggers = list(loggers)
    def log(self, step: int, scalars: Dict[str, Any]) -> None:
        for lg in self._loggers:
            lg.log(step, scalars)
    def flush(self) -> None:
        for lg in self._loggers:
            lg.flush()
    def close(self) -> None:
        for lg in self._loggers:
            lg.close()
