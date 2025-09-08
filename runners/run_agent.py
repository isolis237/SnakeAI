from __future__ import annotations
from config import AppConfig

from core.snake_rules import Rules, Config as SnakeCfg
from core.snake_env import SnakeEnv
from rl.schedulers import LinearDecayEpsilon
from rl.dqn_agent import DQNAgent
from rl.trainer import DQNTrainer, TrainHooks
from rl.logging import CSVLogger, TBLogger, MultiLogger

from rl.torch_net import TorchMLPQNet
from rl.ring_buffer import RingBuffer

from rl.async_episode_player import AsyncEpisodePlayer
from rl.hooks_live import LiveRenderHook
from viz.live_viewer import LiveViewer
from runners.agent_hooks import AgentHooks, ALL_KEYS

import torch

def main(cfg: AppConfig):
    # --- Env
    env = SnakeEnv(Rules(SnakeCfg(
        grid_w=cfg.env.grid_w,
        grid_h=cfg.env.grid_h,
        relative_actions=True,
        seed=cfg.dqn.seed
    )))
    obs_shape = env.observation_shape()
    n_actions = env.action_space_n()

    # --- Live view (optional)
    live_hook = None
    if cfg.render.live_view:
        # Sink (pygame window, video writer, etc.) – used only by the player thread
        sink = LiveViewer(
            fps=cfg.render.view_fps,            # playback FPS (not training speed)
            record_dir=cfg.render.record_dir,
            title="Snake — Training",
            cell_px=cfg.render.cell_px,
            grid_lines=cfg.render.grid_lines,
            show_hud=not cfg.render.hide_hud,
        )
        # Player thread that replays *complete* episodes at a steady FPS
        player = AsyncEpisodePlayer(
            sink,
            fps=cfg.render.view_fps,
            max_episodes_buffered=cfg.runner.player_buffer
        )
        # Hook that RECORDS frames during the episode and enqueues at end
        live_hook = LiveRenderHook(
            sink_player=player,
            grid_w=cfg.env.grid_w,
            grid_h=cfg.env.grid_h,
            stream_episode_prob=cfg.runner.stream_ep_prob,
            best_metric=cfg.runner.best_metric,
            rng_seed=cfg.runner.viewer_seed,
            recent_cap=8,      # small reservoir
            min_fill=2,
            target_fill=4,     # keep ~full
            downsample_stride=2  # optional: halve cached frames
        )

    # --- Networks
    q_net = TorchMLPQNet(obs_shape, n_actions, dueling=cfg.dqn.dueling, device=cfg.dqn.device)
    target_net = TorchMLPQNet(obs_shape, n_actions, dueling=cfg.dqn.dueling, device=cfg.dqn.device)

    # --- Replay & epsilon
    replay = RingBuffer(capacity=cfg.dqn.replay_capacity, seed=cfg.dqn.seed)
    eps = LinearDecayEpsilon(cfg.dqn.epsilon_start, cfg.dqn.epsilon_end, cfg.dqn.epsilon_decay_steps)

    # --- Optimizer
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.dqn.lr)

    # --- Agent & Trainer
    agent = DQNAgent(q_net, target_net, replay, eps, cfg.dqn, optimizer)
    agent.sync_target_soft()

    print("=== Snake DQN ===")
    print(f"grid: {cfg.env.grid_w}x{cfg.env.grid_h}  actions: {n_actions}  device: {cfg.dqn.device}")
    print(f"episodes: {cfg.runner.episodes}  batch: {cfg.dqn.batch_size}  lr: {cfg.dqn.lr}")
    print(f"replay cap: {cfg.dqn.replay_capacity}  warmup: {cfg.dqn.min_replay_before_learn}  dueling: {cfg.dqn.dueling}")
    print("logs: runs/snake_dqn/logs.csv  tb: runs/snake_dqn/tb")

    csv_logger = CSVLogger("runs/snake_dqn/logs.csv", fieldnames=ALL_KEYS)
    try:
        tb_logger = TBLogger("runs/snake_dqn/tb")
        logger = MultiLogger(csv_logger, tb_logger)
    except RuntimeError:
        logger = MultiLogger(csv_logger)

    hooks = AgentHooks(cfg, logger, agent)

    trainer = DQNTrainer(
        env,
        agent,
        TrainHooks(on_step_log=hooks.on_step_log, on_episode_end=hooks.on_episode_end),
        max_steps_per_episode=(cfg.env.max_ep_steps or None),
        live_hook=live_hook
    )

    trainer.train_and_stream(num_episodes=cfg.runner.episodes)
    logger.close()



if __name__ == "__main__":
    # A default AppConfig would be needed to run this file directly.
    pass