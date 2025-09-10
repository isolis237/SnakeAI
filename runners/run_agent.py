from __future__ import annotations
import argparse

from config import AppConfig
from core.snake_rules import Rules

from core.snake_env import SnakeEnv
from rl.dqn_agent import DQNAgent
from rl.trainer import DQNTrainer, TrainHooks
from rl.logging import CSVLogger, TBLogger, MultiLogger
from rl.metrics import EMA, WindowedStat

from rl.torch_net import TorchMLPQNet
from rl.ring_buffer import RingBuffer
from rl.checkpoint import Checkpointer

from rl.utils import resolve_device, LinearDecayEpsilon

from rl.async_episode_player import AsyncEpisodePlayer
from viz.live_viewer import LiveViewer

import torch

from rl.playback import EpisodeRecorder, EpisodeQueuePlayer, LiveHook

ALL_KEYS = [
    "step",
    "train/loss","train/epsilon","train/replay_size","train/q_mean","train/q_max","train/q_min",
    "train/updates","train/grad_norm",
    "train/learn_started","train/replay_fill",        
    "epis/reward","epis/reward_ema","epis/reward_mean100",
    "epis/len","epis/len_ema","epis/len_mean100",
    "epis/final_score","epis/death_wall","epis/death_self","epis/death_starvation",
]


def main():
    cfg = AppConfig()
    rules = Rules(cfg)

    cfg.with_(relative_actions=True)
    env = SnakeEnv(cfg, rules)

    obs_shape = env.observation_shape()
    n_actions = env.action_space_n()

    # --- Live view (optional)
    live_hook = None
    if cfg.live_view:
        sink = LiveViewer(
            fps=cfg.fps,
            record_dir=cfg.render_record_dir,
            title="Snake — Training",
            cell_px=cfg.render_cell,
            grid_lines=cfg.render_grid_lines,
            show_hud=cfg.render_show_hud,
            queue_max=8,
        )
        player = EpisodeQueuePlayer(sink=sink, fps=cfg.fps, max_queue=8)
        rec    = EpisodeRecorder(stream_prob=cfg.stream_ep_prob, rng_seed=cfg.seed)

        live_hook = LiveHook(
            player=player,
            recorder=rec,
            grid_w=cfg.grid_w,
            grid_h=cfg.grid_h,
            keep_best_by=cfg.best_metric,   # "reward" or "final_score"
        )

    # --- Networks
    q_net = TorchMLPQNet(obs_shape, n_actions, dueling=cfg.dueling, device=cfg.device)
    target_net = TorchMLPQNet(obs_shape, n_actions, dueling=cfg.dueling, device=cfg.device)

    # --- Replay & epsilon
    replay = RingBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)
    eps = LinearDecayEpsilon(cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_steps)

    # --- Optimizer
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

    # --- Agent & Trainer
    agent = DQNAgent(q_net, target_net, replay, eps, cfg, optimizer)
    agent.sync_target_soft()

    ckpt = Checkpointer(
        agent,
        out_dir="runs/snake_dqn",
        tag="dqn",
        save_every_episodes=50,                # tweak or None
        best_metric_key="epis/reward_mean100", # or "epis/reward" for faster dev
    )

    print("=== Snake DQN ===")
    print(f"grid: {cfg.grid_w}x{cfg.grid_h}  actions: {n_actions}  device: {cfg.device}")
    print(f"episodes: {cfg.episodes}  batch: {cfg.batch_size}  lr: {cfg.lr}")
    print(f"replay cap: {cfg.replay_capacity}  warmup: {cfg.warmup}  dueling: {cfg.dueling}")
    print("logs: runs/snake_dqn/logs.csv  tb: runs/snake_dqn/tb")

    logger = CSVLogger("runs/snake_dqn/logs.csv", fieldnames=ALL_KEYS)

    ema_reward = EMA(0.05); ema_length = EMA(0.05)
    win_reward = WindowedStat(100); win_length = WindowedStat(100)

    def on_step_log(stats):
        # Log every N updates; but also log occasionally even with no learning
        if stats.get("updates", 0) % 10 != 0 and stats.get("step", 0) % 200 != 0:
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
            "train/learn_started": 1.0 if stats.get("replay_size", 0) >= cfg.warmup else 0.0,
            "train/replay_fill": stats.get("replay_size"),
        }
        logger.log(stats["step"], scalars)

    def on_episode_end(ep, s):
        step = agent.state.global_step
        er, el = s["reward"], s["steps"]
        r_ema = ema_reward.update(er); l_ema = ema_length.update(el)
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
        logger.log(step, scalars)
        logger.flush()

        # --- NEW: checkpointing ---
        ckpt.maybe_save_periodic(ep)
        ckpt.maybe_save_best(scalars)


    trainer = DQNTrainer(
        env,
        agent,
        TrainHooks(on_step_log=on_step_log, on_episode_end=on_episode_end),
        max_steps_per_episode=(cfg.max_ep_steps or None),
        live_hook=live_hook
    )

    # Train + stream/record full episodes (decoupled playback if async hook is used)
    trainer.train_and_stream(num_episodes=cfg.episodes)
    ckpt.save_final()
    logger.close()
