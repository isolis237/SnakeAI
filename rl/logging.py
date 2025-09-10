from __future__ import annotations
import csv, os, time
from typing import Dict, Any, Protocol, Optional, Callable

ALL_KEYS = [
    "step",
    # train
    "train/loss", "train/epsilon", "train/replay_size",
    "train/q_mean", "train/q_max", "train/q_min", "train/updates", "train/grad_norm",
    "train/learn_started", "train/replay_fill",
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

def make_step_logger(
    logger: Logger,
    warmup: int,
    log_every_updates: int = 10,
    also_every_steps: int = 200,
) -> Callable[[dict], None]:
    """
    Returns a function(stats: Dict[str, Any]) -> None that logs training scalars
    at a controlled cadence. Decouples 'main' from logging policy.
    """
    def _on_step_log(stats: Dict[str, Any]) -> None:
        # “cadence” rule: every N updates, and occasionally every M steps
        if stats.get("updates", 0) % log_every_updates != 0 and stats.get("step", 0) % also_every_steps != 0:
            return

        scalars = {
            "train/loss": stats.get("loss"),
            "train/epsilon": stats.get("epsilon"),
            "train/replay_size": stats.get("replay_size"),
            "train/q_mean": stats.get("q_mean"),
            "train/q_max": stats.get("q_max"),
            "train/q_min": stats.get("q_min"),
            "train/updates": stats.get("updates"),
            "train/grad_norm": stats.get("grad_norm"),
            "train/learn_started": 1.0 if stats.get("replay_size", 0) >= warmup else 0.0,
            "train/replay_fill": stats.get("replay_size"),
        }
        # guard: require 'step' to be present for CSV ‘step’ column
        step = stats.get("step")
        if step is None:
            return
        logger.log(int(step), scalars)
    return _on_step_log

def make_episode_logger(
    *,
    logger: Logger,
    ema_reward,
    ema_length,
    win_reward,
    win_length,
    step_getter: Callable[[], int],
    # Optional checkpoint hooks to avoid importing Checkpointer here:
    ckpt_periodic: Optional[Callable[[int], None]] = None,
    ckpt_best: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """
    Returns a function(ep: int, s: Dict[str, Any]) -> None that:
      - updates EMA/window stats
      - logs all episode metrics
      - optionally triggers checkpoint hooks

    'step_getter' should return the global training step for the CSV 'step' column.
    """
    def _on_episode_end(ep: int, s: Dict[str, Any]) -> None:
        er, el = float(s["reward"]), int(s["steps"])
        r_ema = ema_reward.update(er)
        l_ema = ema_length.update(el)

        win_reward.add(er); win_length.add(el)
        wr, wl = win_reward.summary(), win_length.summary()

        scalars = {
            "episode": ep,
            "epis/reward": er,
            "epis/reward_ema": r_ema,
            "epis/reward_mean100": wr["mean"],
            "epis/len": el,
            "epis/len_ema": l_ema,
            "epis/len_mean100": wl["mean"],
            "epis/final_score": s.get("final_score", 0),
            "epis/death_wall": 1.0 if s.get("death_reason") == "wall" else 0.0,
            "epis/death_self": 1.0 if s.get("death_reason") == "self" else 0.0,
            "epis/death_starvation": 1.0 if s.get("death_reason") == "starvation" else 0.0,
        }

        step = int(step_getter())
        logger.log(step, scalars)
        logger.flush()

        if ckpt_periodic is not None:
            ckpt_periodic(ep)
        if ckpt_best is not None:
            ckpt_best(scalars)

    return _on_episode_end