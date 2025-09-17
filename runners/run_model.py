# runners/run_model.py
from __future__ import annotations

import os
from typing import Optional

import torch

from config import AppConfig
from core.snake_rules import Rules
from core.snake_env import SnakeEnv

from rl.torch_net import TorchMLPQNet
from rl.torch_cnn_qnet import TorchCNNQNet

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

    for t in candidates:
        if os.path.exists(os.path.join(ckpt_dir, f"{t}_online.pt")):
            return t

    for name in sorted(os.listdir(ckpt_dir)):
        if name.endswith("_online.pt"):
            return name[: -len("_online.pt")]

    raise FileNotFoundError(f"No '*_online.pt' checkpoint found in {ckpt_dir}")


def _infer_model_type(ckpt_dir: str, model_hint: Optional[str]) -> str:
    """
    Decide which net to build. If model_hint is supplied, trust it.
    Otherwise infer from directory name (cnn vs mlp).
    """
    if model_hint in ("mlp", "cnn"):
        return model_hint
    d = ckpt_dir.replace("\\", "/").lower()
    if d.endswith("/cnn") or "/cnn/" in d:
        return "cnn"
    return "mlp"


def _build_qnets(model_type: str, obs_shape, n_actions, dueling: bool, device: str):
    """
    Return (q_net, target_net) of the appropriate architecture.
    """
    if model_type == "cnn":
        if TorchCNNQNet is None:
            raise RuntimeError(
                "CNN checkpoints provided but TorchCNNQNet is not available/importable. "
                "Adjust the import path or set model_hint='mlp' if these weights are actually MLP."
            )
        q = TorchCNNQNet(obs_shape, n_actions, dueling=dueling, device=device)
        t = TorchCNNQNet(obs_shape, n_actions, dueling=dueling, device=device)
    else:
        q = TorchMLPQNet(obs_shape, n_actions, dueling=dueling, device=device)
        t = TorchMLPQNet(obs_shape, n_actions, dueling=dueling, device=device)
    return q, t


def run_single(
    *,
    episodes: int = 1,
    ckpt_dir: str = "runs/snake_dqn/mlp",
    ckpt_prefer: str = "best",
    title: Optional[str] = None,
    model_hint: Optional[str] = None,  # "mlp" | "cnn" | None (infer from path)
) -> None:
    """
    Internal helper used by compare_models and by this module's main().
    Runs a single viewer for a single checkpoint directory.
    """
    # Config & FPS (honor env override)
    cfg = AppConfig().with_(relative_actions=True)
    cfg = cfg.with_(fps=18)
    if title is not None:
        cfg = cfg.with_(render_title=title)

    rules = Rules(cfg)
    env = SnakeEnv(cfg, rules)
    obs_shape = env.observation_shape()
    n_actions = env.action_space_n()

    # Pick architecture and build nets
    model_type = _infer_model_type(ckpt_dir, model_hint)
    q_net, target_net = _build_qnets(model_type, obs_shape, n_actions, cfg.dueling, cfg.device)

    # Agent (greedy eval)
    replay = RingBuffer(capacity=1, seed=cfg.seed)
    eps = LinearDecayEpsilon(0.0, 0.0, 1)  # ε=0 → greedy
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    agent = DQNAgent(q_net, target_net, replay, eps, cfg, optimizer)

    # Load weights
    used_tag = _pick_tag_from_dir(ckpt_dir, prefer=ckpt_prefer, base="dqn")
    agent.load_weights_only(ckpt_dir, used_tag)
    target_net.load_state_dict(q_net.state_dict(), strict=False)

    print(f"[model] loaded checkpoint tag: {used_tag}")
    print(f"[model] arch={model_type}  relative_actions={cfg.relative_actions}  n_actions={n_actions}  fps={cfg.fps}")

    q_net.eval()
    target_net.eval()

    # Live viewer
    sink = LiveViewer(cfg)
    player = EpisodeQueuePlayer(sink=sink, fps=cfg.fps, max_queue=cfg.queue_max)
    rec = EpisodeRecorder()
    live = LiveHook(
        player=player,
        recorder=rec,
        cfg=cfg,
        stream_prob=1.0,  # stream every episode
        recent_cap=0,
        min_fill=0,
        target_fill=0,
    )
    live.ensure_started()

    # Rollout episodes
    for ep in range(episodes):
        live.start_episode(ep)
        obs = env.reset()

        # initial snapshot
        try:
            live.record_snapshot(env.get_snapshot())
        except Exception:
            pass

        done = False
        steps = 0
        ep_reward = 0.0
        last_info = {}

        with torch.no_grad():
            while not done:
                a = agent.act(obs)
                step = env.step(a)
                last_info = step.info or {}

                # per-step snapshot for smooth playback
                try:
                    live.record_snapshot(env.get_snapshot())
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

    # Wait for the player to finish rendering frames before closing
    player.wait_idle(timeout=25.0)

    live.close()


def main() -> None:
    """
    No-arg entrypoint to match your top-level main.py pattern.
    Defaults:
      - episodes: 10
      - ckpt_dir: runs/snake_dqn/mlp
      - ckpt_prefer: 'best'
      - fps: honor SNAKE_FPS if set; otherwise AppConfig default
    To run CNN instead, either:
      - set SNAKE_CKPT_DIR=runs/snake_dqn/cnn (recommended), or
      - adjust the default below if you prefer CNN as the default.
    """
    ckpt_dir = os.getenv("SNAKE_CKPT_DIR", "runs/snake_dqn/cnn")
    title = "cnn"

    # If user switches dir to cnn, we’ll infer cnn automatically.
    run_single(
        episodes=1,
        ckpt_dir=ckpt_dir,
        ckpt_prefer="best",
        title=title,
        model_hint="cnn",  # "mlp" | "cnn" | None (infer from path)
    )
