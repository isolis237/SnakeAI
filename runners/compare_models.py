# runners/compare_models.py
from __future__ import annotations

import os
from typing import Tuple

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

# Reuse your A* policy
from runners.run_astar import AStarPolicy


def _pick_tag_from_dir(ckpt_dir: str, prefer: str = "best", base: str = "dqn") -> str:
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


def _build_agent(model: str, obs_shape, n_actions, cfg: AppConfig) -> DQNAgent:
    if model == "mlp":
        q = TorchMLPQNet(obs_shape, n_actions, dueling=cfg.dueling, device=cfg.device)
        t = TorchMLPQNet(obs_shape, n_actions, dueling=cfg.dueling, device=cfg.device)
    elif model == "cnn":
        q = TorchCNNQNet(obs_shape, n_actions, dueling=cfg.dueling, device=cfg.device)
        t = TorchCNNQNet(obs_shape, n_actions, dueling=cfg.dueling, device=cfg.device)
    else:
        raise ValueError(f"unknown model type: {model}")

    replay = RingBuffer(capacity=1, seed=cfg.seed)
    eps = LinearDecayEpsilon(0.0, 0.0, 1)  # greedy eval
    optimizer = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    agent = DQNAgent(q, t, replay, eps, cfg, optimizer)

    # sync target; eval mode
    t.load_state_dict(q.state_dict(), strict=False)
    q.eval()
    t.eval()
    return agent


def _eval_n_episodes_model(
    *,
    env: SnakeEnv,
    live: LiveHook,
    agent: DQNAgent,
    n_episodes: int,
    cfg: AppConfig,
) -> Tuple[float, int]:
    """Evaluate a learned agent without streaming; LiveHook keeps the best."""
    best_metric = float("-inf")
    last_steps = 0

    for ep in range(n_episodes):
        live.start_episode(ep)
        obs = env.reset()

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

        metric = float(summary.get("final_score", summary.get("reward", 0.0)))
        best_metric = max(best_metric, metric)
        last_steps = steps

        print(
            f"[compare-3] model ep {ep:04d}  score={summary['final_score']}  "
            f"reward={summary['reward']:.2f}  steps={steps}"
        )

    return best_metric, last_steps


def _eval_n_episodes_astar(
    *,
    env: SnakeEnv,
    live: LiveHook,
    policy: AStarPolicy,
    n_episodes: int,
    cfg_for_astar: AppConfig,
) -> Tuple[float, int]:
    """Evaluate A* with its own action mode (absolute)."""
    best_metric = float("-inf")
    last_steps = 0

    for ep in range(n_episodes):
        live.start_episode(ep)
        obs = env.reset()

        try:
            live.record_snapshot(env.get_snapshot())
        except Exception:
            pass

        done = False
        steps = 0
        ep_reward = 0.0

        while not done:
            snap = env.get_snapshot()

            try:
                live.record_snapshot(snap)
            except Exception:
                pass

            # IMPORTANT: Tell the policy which action convention to use
            policy.set_snapshot(snap, relative_actions=cfg_for_astar.relative_actions)

            a = policy.act(obs)
            res = env.step(a)
            obs = res.obs
            done = res.terminated or res.truncated
            steps += 1
            ep_reward += float(res.reward)

            if cfg_for_astar.max_ep_steps and steps >= cfg_for_astar.max_ep_steps:
                break

        final = env.get_snapshot()
        summary = {
            "steps": steps,
            "reward": ep_reward,
            "final_score": final.score,
            "death_reason": final.reason,
        }
        live.end_episode(summary, {})

        metric = float(summary.get("final_score", summary.get("reward", 0.0)))
        best_metric = max(best_metric, metric)
        last_steps = steps

        print(
            f"[compare-3] A* ep {ep:04d}  score={summary['final_score']}  "
            f"reward={summary['reward']:.2f}  steps={steps}"
        )

    return best_metric, last_steps


def main() -> None:
    # Inputs
    episodes = int(os.getenv("SNAKE_COMPARE_EPISODES", "15"))
    mlp_dir = os.getenv("SNAKE_MLP_DIR", "runs/snake_dqn/mlp")
    cnn_dir = os.getenv("SNAKE_CNN_DIR", "runs/snake_dqn/cnn")

    # Learned models use RELATIVE actions (3-way)
    cfg_rel = AppConfig().with_(relative_actions=True)
    cfg_rel = cfg_rel.with_(fps=int(os.getenv("SNAKE_FPS", cfg_rel.fps)))

    # A* uses ABSOLUTE actions (4-way), matching your standalone runner
    cfg_abs = cfg_rel.with_(relative_actions=False)

    # Probe shapes once
    rules_probe = Rules(cfg_rel)
    env_probe = SnakeEnv(cfg_rel, rules_probe)
    obs_shape = env_probe.observation_shape()
    n_actions_rel = env_probe.action_space_n()

    # ======== MLP collect (relative) ========
    rules_mlp = Rules(cfg_rel)
    env_mlp = SnakeEnv(cfg_rel, rules_mlp)
    agent_mlp = _build_agent("mlp", obs_shape, n_actions_rel, cfg_rel)
    tag_mlp = _pick_tag_from_dir(mlp_dir, prefer="best", base="dqn")
    agent_mlp.load_weights_only(mlp_dir, tag_mlp)

    cfg_mlp = cfg_rel.with_(render_title="MLP — Best Replay")
    sink_mlp = LiveViewer(cfg_mlp)
    player_mlp = EpisodeQueuePlayer(sink=sink_mlp, fps=cfg_rel.fps, max_queue=cfg_rel.queue_max)
    live_mlp = LiveHook(player=player_mlp, recorder=EpisodeRecorder(), cfg=cfg_rel,
                        stream_prob=0.0, recent_cap=0, min_fill=0, target_fill=0)
    live_mlp.ensure_started()

    print(f"[compare-3] MLP from {mlp_dir}  tag={tag_mlp}  fps={cfg_rel.fps}")
    mlp_best, _ = _eval_n_episodes_model(env=env_mlp, live=live_mlp, agent=agent_mlp,
                                         n_episodes=episodes, cfg=cfg_rel)

    # ======== CNN collect (relative) ========
    rules_cnn = Rules(cfg_rel)
    env_cnn = SnakeEnv(cfg_rel, rules_cnn)
    agent_cnn = _build_agent("cnn", obs_shape, n_actions_rel, cfg_rel)
    tag_cnn = _pick_tag_from_dir(cnn_dir, prefer="best", base="dqn")
    agent_cnn.load_weights_only(cnn_dir, tag_cnn)

    cfg_cnn = cfg_rel.with_(render_title="CNN — Best Replay")
    sink_cnn = LiveViewer(cfg_cnn)
    player_cnn = EpisodeQueuePlayer(sink=sink_cnn, fps=cfg_rel.fps, max_queue=cfg_rel.queue_max)
    live_cnn = LiveHook(player=player_cnn, recorder=EpisodeRecorder(), cfg=cfg_rel,
                        stream_prob=0.0, recent_cap=0, min_fill=0, target_fill=0)
    live_cnn.ensure_started()

    print(f"[compare-3] CNN from {cnn_dir}  tag={tag_cnn}  fps={cfg_rel.fps}")
    cnn_best, _ = _eval_n_episodes_model(env=env_cnn, live=live_cnn, agent=agent_cnn,
                                         n_episodes=episodes, cfg=cfg_rel)

    # ======== A* collect (absolute) ========
    rules_astar = Rules(cfg_abs)
    env_astar = SnakeEnv(cfg_abs, rules_astar)
    policy = AStarPolicy(consider_tail_free=True)

    cfg_astar_view = cfg_abs.with_(render_title="A* — Best Replay")
    sink_astar = LiveViewer(cfg_astar_view)
    player_astar = EpisodeQueuePlayer(sink=sink_astar, fps=cfg_abs.fps, max_queue=cfg_abs.queue_max)
    live_astar = LiveHook(player=player_astar, recorder=EpisodeRecorder(), cfg=cfg_abs,
                          stream_prob=0.0, recent_cap=0, min_fill=0, target_fill=0)
    live_astar.ensure_started()

    print(f"[compare-3] A* policy (absolute actions)  fps={cfg_abs.fps}")
    astar_best, _ = _eval_n_episodes_astar(env=env_astar, live=live_astar, policy=policy,
                                           n_episodes=episodes, cfg_for_astar=cfg_abs)

    # ======== Play all three best episodes concurrently ========
    print("[compare-3] Playing best episodes for MLP, CNN, and A* concurrently…")

    # Enqueue best on each window without blocking
    live_mlp.replay_best_and_wait(timeout=0.0)
    live_cnn.replay_best_and_wait(timeout=0.0)
    live_astar.replay_best_and_wait(timeout=0.0)

    # Wait for each to finish
    player_mlp.wait_idle()
    player_cnn.wait_idle()
    player_astar.wait_idle()

    # Close viewers
    live_mlp.close()
    live_cnn.close()
    live_astar.close()

    # Summary
    print("\n[compare-3] ===== Summary (best-of-{0}) =====".format(episodes))
    print(f"[compare-3] MLP best metric: {mlp_best:.2f}")
    print(f"[compare-3] CNN best metric: {cnn_best:.2f}")
    print(f"[compare-3] A*  best metric: {astar_best:.2f}")
    if cnn_best > mlp_best:
        print("[compare-3] Learned winner: CNN 🎉")
    elif mlp_best > cnn_best:
        print("[compare-3] Learned winner: MLP 🎉")
    else:
        print("[compare-3] Learned models tied.")
