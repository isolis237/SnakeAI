# runners/compare_models.py
from __future__ import annotations
import multiprocessing as mp
from runners.run_model import main as run_one

def _proc(prefer: str, title: str, episodes: int, ckpt_dir: str):
    # each process runs an independent viewer
    run_one(
        episodes=episodes,
        ckpt_dir=ckpt_dir,
        ckpt_prefer=prefer,
        title=title,
    )

def main(
    *,
    episodes: int = 10,
    ckpt_dir: str = "runs/snake_dqn",
) -> None:
    # What we are comparing (you asked ep1500, ep5000, best)
    jobs = [
        ("ep1500", "Snake — ep1500"),
        ("ep12500", "Snake — ep12500"),
        ("best",   "Snake — BEST"),
    ]

    ctx = mp.get_context("spawn")   # safest with torch + pygame
    procs = []
    for prefer, title in jobs:
        p = ctx.Process(
            target=_proc,
            kwargs=dict(prefer=prefer, title=title, episodes=episodes, ckpt_dir=ckpt_dir),
            daemon=False,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
