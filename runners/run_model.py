# runners/run_model.py
from __future__ import annotations

import os
import torch
from typing import Optional

from config import AppConfig
from core.snake_rules import Rules
from core.snake_env import SnakeEnv

from rl.torch_net import TorchMLPQNet
from rl.ring_buffer import RingBuffer
from rl.utils import LinearDecayEpsilon
from rl.dqn_agent import DQNAgent
from rl.playback import EpisodeRecorder, EpisodeQueuePlayer, LiveHook
from viz.live_viewer import LiveViewer


def _pick_tag_from_dir(ckpt_dir: str, prefer: str = "best", base: str = "dqn") -> str:
    """
    Return the checkpoint TAG (e.g., 'dqn_best', 'dqn_final', 'dqn_ep500') by checking files.
    We look for '<tag>_online.pt' because that's the only thing we truly need for eval.
    """
    candidates = []
    if prefer in ("best", "final"):
        candidates = [f"{base}_{prefer}"]
    elif prefer.startswith("ep"):
        candidates = [f"{base}_{prefer}"]
    else:
        candidates = [f"{base}_best", f"{base}_final"]

    # try preferred candidates first
    for t in candidates:
        if os.path.exists(os.path.join(ckpt_dir, f"{t}_online.pt")):
            return t

    # fallback: find any *_online.pt and derive tag
    for name in sorted(os.listdir(ckpt_dir)):
        if name.endswith("_online.pt"):
            return name[:-len("_online.pt")]  # strip suffix to get the tag

    raise FileNotFoundError(f"No '*_online.pt' checkpoint found in {ckpt_dir}")


def main(
    *,
    episodes: int = 10,
    ckpt_dir: str = "runs/snake_dqn",
    ckpt_prefer: str = "best",          # "best" | "final" | "epXXXX"
    title: Optional[str] = None,         # NEW: optional window title override
) -> None:
    # --- Config & Env
    # Match how you trained (you discovered training used relative_actions=False).
    cfg = AppConfig().with_(relative_actions=True).with_(fps=8)
    if title is not None:
        cfg = cfg.with_(render_title=title)

    rules = Rules(cfg)
    env = SnakeEnv(cfg, rules)

    obs_shape = env.observation_shape()
    n_actions = env.action_space_n()

    # --- Networks (same arch as training)
    q_net = TorchMLPQNet(obs_shape, n_actions, dueling=cfg.dueling, device=cfg.device)
    target_net = TorchMLPQNet(obs_shape, n_actions, dueling=cfg.dueling, device=cfg.device)

    # --- Dummy replay & ε=0 schedule (pure greedy)
    replay = RingBuffer(capacity=1, seed=cfg.seed)   # tiny; unused in eval
    eps = LinearDecayEpsilon(0.0, 0.0, 1)            # always 0 -> greedy

    # --- Optimizer is irrelevant in eval, but agent expects one; LR doesn’t matter
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

    # --- Agent
    agent = DQNAgent(q_net, target_net, replay, eps, cfg, optimizer)

    # load checkpoint weights
    used_tag = _pick_tag_from_dir(ckpt_dir, prefer=ckpt_prefer, base="dqn")
    agent.load_weights_only(ckpt_dir, used_tag)
    # If target wasn't saved (or for safety), mirror online -> target
    target_net.load_state_dict(q_net.state_dict(), strict=False)

    print(f"[model] loaded checkpoint tag (weights-only): {used_tag}")
    print(f"[model] relative_actions={cfg.relative_actions} n_actions={n_actions}")

    q_net.eval(); target_net.eval()

    # --- Live viewer pipeline (always stream episodes during eval)
    sink   = LiveViewer(cfg)
    player = EpisodeQueuePlayer(sink=sink, fps=cfg.fps, max_queue=cfg.queue_max)
    rec    = EpisodeRecorder()  # always capture
    live   = LiveHook(
        player=player,
        recorder=rec,
        cfg=cfg,
        stream_prob=1.0,          # enqueue every finished episode for playback
        recent_cap=0,             # disable feeder (no need in eval)
        min_fill=0, target_fill=0 # ^
    )
    live.ensure_started()

    # --- Run greedy episodes (no learning)
    for ep in range(episodes):
        live.start_episode(ep)

        obs = env.reset()
        # initial snapshot
        try:
            snap0 = env.get_snapshot()
            live.record_snapshot(snap0)
        except Exception:
            pass

        done = False
        steps = 0
        ep_reward = 0.0
        last_info = {}

        with torch.no_grad():
            while not done:
                # greedy action: agent.act respects eps=0 scheduler
                a = agent.act(obs)
                step = env.step(a)
                last_info = step.info or {}

                # capture snapshot every step for smooth playback
                try:
                    snap = env.get_snapshot()
                    live.record_snapshot(snap)
                except Exception:
                    pass

                obs = step.obs
                done = step.terminated or step.truncated
                steps += 1
                ep_reward += float(step.reward)

                if cfg.max_ep_steps and steps >= cfg.max_ep_steps:
                    break

        summary = {
            "steps": steps,
            "reward": ep_reward,
            "final_score": last_info.get("score", 0),
            "death_reason": last_info.get("reason"),
        }
        live.end_episode(summary, last_info)

        print(f"[model] ep {ep:04d}  score={summary['final_score']}  reward={summary['reward']:.2f}  steps={steps}")

    # encore the best and wait for it to finish
    live.replay_best_and_wait(timeout=20.0)
    live.close()
