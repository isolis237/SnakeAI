from __future__ import annotations
import argparse

from core.snake_rules import Rules, Config as SnakeCfg
from core.snake_env import SnakeEnv
from rl.dqn_config import DQNConfig
from rl.schedulers import LinearDecayEpsilon
from rl.dqn_agent import DQNAgent
from rl.trainer import DQNTrainer, TrainHooks
from rl.logging import CSVLogger, TBLogger, MultiLogger
from rl.metrics import EMA, WindowedStat

from rl.torch_net import TorchMLPQNet
from rl.ring_buffer import RingBuffer

from rl.utils import resolve_device

from rl.async_episode_player import AsyncEpisodePlayer
from rl.hooks_live import LiveRenderHook  # the async/episode-queued version
from viz.live_viewer import LiveViewer

import torch

ALL_KEYS = [
    "step",
    "train/loss","train/epsilon","train/replay_size","train/q_mean","train/q_max","train/q_min",
    "train/updates","train/grad_norm",
    "train/learn_started","train/replay_fill",           # ← NEW
    "epis/reward","epis/reward_ema","epis/reward_mean100",
    "epis/len","epis/len_ema","epis/len_mean100",
    "epis/final_score","epis/death_wall","epis/death_self","epis/death_starvation",
]


def main(args):
    # --- Env
    env = SnakeEnv(Rules(SnakeCfg(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        relative_actions=True,
        seed=args.seed
    )))
    obs_shape = env.observation_shape()
    n_actions = env.action_space_n()

    # --- Live view (optional)
    live_hook = None
    if args.live_view:
        # Sink (pygame window, video writer, etc.) – used only by the player thread
        sink = LiveViewer(
            fps=args.view_fps,            # playback FPS (not training speed)
            record_dir=args.record_dir,
            title="Snake — Training",
            cell_px=args.cell_px,
            grid_lines=args.grid_lines,
            show_hud=not args.hide_hud,
        )
        # Player thread that replays *complete* episodes at a steady FPS
        player = AsyncEpisodePlayer(
            sink,
            fps=args.view_fps,
            max_episodes_buffered=args.player_buffer
        )
        # Hook that RECORDS frames during the episode and enqueues at end
        live_hook = LiveRenderHook(
            sink_player=player,
            grid_w=args.grid_w,
            grid_h=args.grid_h,
            stream_episode_prob=args.stream_ep_prob,
            best_metric=args.best_metric,
            rng_seed=args.viewer_seed,
        )

    # --- Networks
    q_net = TorchMLPQNet(obs_shape, n_actions, dueling=args.dueling, device=args.device)
    target_net = TorchMLPQNet(obs_shape, n_actions, dueling=args.dueling, device=args.device)

    # --- Replay & epsilon
    replay = RingBuffer(capacity=args.cap, seed=args.seed)
    eps = LinearDecayEpsilon(DQNConfig.epsilon_start, DQNConfig.epsilon_end, DQNConfig.epsilon_decay_steps)

    # --- Config & optimizer
    cfg = DQNConfig(
        batch_size=args.batch,
        lr=args.lr,
        min_replay_before_learn=args.warmup,
        device=args.device,
        double_dqn=True,
        dueling=args.dueling,
    )
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

    # --- Agent & Trainer
    agent = DQNAgent(q_net, target_net, replay, eps, cfg, optimizer)
    agent.sync_target_hard()

    print("=== Snake DQN ===")
    print(f"grid: {args.grid_w}x{args.grid_h}  actions: {n_actions}  device: {args.device}")
    print(f"episodes: {args.episodes}  batch: {args.batch}  lr: {args.lr}")
    print(f"replay cap: {args.cap}  warmup: {args.warmup}  dueling: {args.dueling}")
    print("logs: runs/snake_dqn/logs.csv  tb: runs/snake_dqn/tb")

    csv_logger = CSVLogger("runs/snake_dqn/logs.csv", fieldnames=ALL_KEYS)
    try:
        tb_logger = TBLogger("runs/snake_dqn/tb")
        logger = MultiLogger(csv_logger, tb_logger)
    except RuntimeError:
        logger = MultiLogger(csv_logger)

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
            "train/learn_started": 1.0 if stats.get("replay_size", 0) >= args.warmup else 0.0,
            "train/replay_fill": stats.get("replay_size"),
        }
        logger.log(stats["step"], scalars)

        print(f"[step {stats['step']:>7}] upd={stats['updates']:>6} "
            f"loss={scalars['train/loss'] if scalars['train/loss'] is not None else float('nan'):.4f} "
            f"eps={scalars['train/epsilon']:.3f} "
            f"replay={scalars['train/replay_size']:>6} "
            f"learn_started={int(scalars['train/learn_started'])} "
            f"qμ={scalars['train/q_mean']:.3f} qmax={scalars['train/q_max']:.3f}")


    def on_episode_end(ep, s):
        step = agent.state.global_step
        er, el = s["reward"], s["steps"]
        r_ema = ema_reward.update(er); l_ema = ema_length.update(el)
        win_reward.add(er); win_length.add(el)
        wr, wl = win_reward.summary(), win_length.summary()
        logger.log(step, {
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
        })
        logger.flush()

        print(f"[episode {ep:>5}] step={step:>7}  R={er:7.3f} (EMA {r_ema:7.3f})  "
              f"len={el:4d} (EMA {l_ema:7.2f})  score={s.get('final_score',0)}  "
              f"death={s.get('death_reason')}")

    trainer = DQNTrainer(
        env,
        agent,
        TrainHooks(on_step_log=on_step_log, on_episode_end=on_episode_end),
        max_steps_per_episode=(args.max_ep_steps or None),
        live_hook=live_hook
    )

    # Train + stream/record full episodes (decoupled playback if async hook is used)
    trainer.train_and_stream(num_episodes=args.episodes)
    logger.close()



if __name__ == "__main__":
    main()