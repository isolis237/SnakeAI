# runners/run_model.py
from __future__ import annotations
import argparse

from core.snake_rules import Rules as SnakeCfg
from core.snake_env import SnakeEnv

from rl.torch_net import TorchMLPQNet
from rl.ring_buffer import RingBuffer
from rl.schedulers import LinearDecayEpsilon
from rl.dqn_config import DQNConfig
from rl.dqn_agent import DQNAgent

from rl.async_episode_player import AsyncEpisodePlayer
from rl.hooks_live import LiveRenderHook
from viz.live_viewer import LiveViewer

import torch

def _build_env(args):
    return SnakeEnv(Rules(SnakeCfg(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        relative_actions=True,
        seed=args.seed,
    )))

def _build_agent(env, args):
    obs_shape = env.observation_shape()
    n_actions = env.action_space_n()

    q_net = TorchMLPQNet(obs_shape, n_actions, dueling=args.dueling, device=args.device)
    target_net = TorchMLPQNet(obs_shape, n_actions, dueling=args.dueling, device=args.device)

    # Tiny shells just to satisfy ctor; never used in playback
    replay = RingBuffer(capacity=1, seed=args.seed)
    from rl.schedulers import LinearDecayEpsilon
    eps = LinearDecayEpsilon(0.0, 0.0, 1)

    from rl.dqn_config import DQNConfig
    cfg = DQNConfig(
        batch_size=1, lr=1e-4, min_replay_before_learn=10**9,
        device=args.device, double_dqn=True, dueling=args.dueling,
    )

    import torch
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

    agent = DQNAgent(q_net, target_net, replay, eps, cfg, optimizer)
    # **Weights only** – avoids touching replay/rng/state
    agent.load_weights_only(args.ckpt_dir, tag=args.tag)
    agent.eval_mode()
    return agent


def play_once(env, agent, live_hook=None, max_steps=None):
    # 1) Reset FIRST so get_snapshot() is valid
    obs = env.reset()

    # 2) Only now boot the viewer and announce the episode
    if live_hook is not None:
        live_hook.ensure_started()
        live_hook.on_episode_start(0)

        # 3) After on_episode_start(), streaming is decided; now we can check
        if live_hook.is_streaming():
            try:
                live_hook.collect_snapshot(env.get_snapshot())  # initial frame
            except Exception:
                pass

    done, steps, total_r = False, 0, 0.0
    last_info = {}

    while not done:
        action = agent.act_greedy(obs)
        step = env.step(action)
        last_info = step.info or last_info

        if live_hook is not None and live_hook.is_streaming():
            try:
                live_hook.collect_snapshot(env.get_snapshot())
            except Exception:
                pass

        obs = step.obs
        done = step.terminated or step.truncated
        steps += 1
        total_r += float(step.reward)
        if max_steps and steps >= max_steps:
            break

    if live_hook is not None:
        live_hook.on_episode_end({
            "steps": steps, "reward": total_r,
            "final_score": last_info.get("score", 0),
            "death_reason": last_info.get("reason"),
        }, last_info)
        # Optional to show the episode once more
        try:
            live_hook.replay_best()
        except Exception:
            pass
        live_hook.close()

    return {"steps": steps, "reward": total_r, "info": last_info}

def _maybe_make_live_hook(args):
    if not args.live_view:
        return None
    sink = LiveViewer(
        fps=args.view_fps,
        record_dir=args.record_dir,
        title=f"Snake — Playback ({args.tag})",
        cell_px=args.cell_px,
        grid_lines=args.grid_lines,
        show_hud=True,
    )
    player = AsyncEpisodePlayer(sink, fps=args.view_fps, max_episodes_buffered=2)
    return LiveRenderHook(
        sink_player=player,
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        stream_episode_prob=1.0,  # always stream in playback
        best_metric="reward",
        rng_seed=args.seed,
        recent_cap=1, min_fill=1, target_fill=1, downsample_stride=1
    )

def main(args):
    env = _build_env(args)
    agent = _build_agent(env, args)
    live_hook = _maybe_make_live_hook(args)

    for i in range(args.play_episodes):
        res = play_once(env, agent, live_hook=live_hook)
        print(f"[Playback {i}] reward={res['reward']:.2f} steps={res['steps']} info={res['info']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # keep arg parsing in top-level main.py; this file is importable
    pass
