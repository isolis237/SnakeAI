from __future__ import annotations

import torch

from config import AppConfig
from core.snake_rules import Rules
from core.snake_env import SnakeEnv

from rl.dqn_agent import DQNAgent
from rl.trainer import DQNTrainer, TrainHooks
from rl.logging import CSVLogger, make_step_logger, make_episode_logger, ALL_KEYS
from rl.metrics import EMA, WindowedStat
from rl.torch_net import TorchMLPQNet
from rl.ring_buffer import RingBuffer
from rl.checkpoint import Checkpointer
from rl.utils import LinearDecayEpsilon
from rl.playback import EpisodeRecorder, EpisodeQueuePlayer, LiveHook

from viz.live_viewer import LiveViewer


def main():
    # Init Config and Env
    cfg = AppConfig()
    rules = Rules(cfg)

    cfg.with_(relative_actions=True)
    env = SnakeEnv(cfg, rules)

    obs_shape = env.observation_shape()
    n_actions = env.action_space_n()

    # --- Live view
    live_hook = None
    if cfg.live_view:
        sink = LiveViewer(cfg)
        player = EpisodeQueuePlayer(sink=sink, fps=cfg.fps, max_queue=cfg.queue_max)
        rec    = EpisodeRecorder(stream_prob=cfg.stream_ep_prob, rng_seed=cfg.seed)
        live_hook = LiveHook(player=player, recorder=rec, cfg=cfg)

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

    # --- Checkpointer
    ckpt = Checkpointer(agent, out_dir="runs/snake_dqn", tag="dqn",
                        save_every_episodes=500, best_metric_key="epis/reward_mean100")

    print("=== Snake DQN ===")
    print(f"grid: {cfg.grid_w}x{cfg.grid_h}  actions: {n_actions}  device: {cfg.device}")
    print(f"episodes: {cfg.episodes}  batch: {cfg.batch_size}  lr: {cfg.lr}")
    print(f"replay cap: {cfg.replay_capacity}  warmup: {cfg.warmup}  dueling: {cfg.dueling}")
    print("logs: runs/snake_dqn/logs.csv  tb: runs/snake_dqn/tb")

    logger = CSVLogger("runs/snake_dqn/logs.csv", fieldnames=ALL_KEYS)
    on_step_log = make_step_logger(logger=logger, warmup=cfg.warmup, 
                                    log_every_updates=10, also_every_steps=200)

    on_episode_end = make_episode_logger(
        logger=logger, ema_reward=EMA(0.05), ema_length=EMA(0.05),
        win_reward=WindowedStat(100), win_length=WindowedStat(100), 
        step_getter=lambda: agent.state.global_step,
        ckpt_periodic=ckpt.maybe_save_periodic, ckpt_best=ckpt.maybe_save_best)

    trainer = DQNTrainer(
        env, agent, TrainHooks(on_step_log=on_step_log, on_episode_end=on_episode_end),
        max_steps_per_episode=(cfg.max_ep_steps or None), live_hook=live_hook)

    # Train + stream/record full episodes (decoupled playback if async hook is used)
    trainer.train_and_stream(num_episodes=cfg.episodes)
    ckpt.save_final()
    logger.close()
