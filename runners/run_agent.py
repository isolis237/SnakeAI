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

import torch

ALL_KEYS = [
    "step",
    # train
    "train/loss", "train/epsilon", "train/replay_size",
    "train/q_mean", "train/q_max", "train/q_min", "train/updates",
    # episode
    "epis/reward", "epis/reward_ema", "epis/reward_mean100",
    "epis/len", "epis/len_ema", "epis/len_mean100",
    "epis/final_score", "epis/death_wall", "epis/death_self", "epis/death_starvation",
]

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["snake", "snake_agent", "loop", "multi", "nnview"])
    p.add_argument("--grid_w", type=int, default=12)
    p.add_argument("--grid_h", type=int, default=12)
    p.add_argument("--episodes", type=int, default=750)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")  # or "auto" if you use resolve_device()
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-35)
    p.add_argument("--warmup", type=int, default=10_000)
    p.add_argument("--cap", type=int, default=150_000)
    p.add_argument("--dueling", action="store_true")
    p.add_argument("--max_ep_steps", type=int, default=0)
    p.add_argument("--log_every_updates", type=int, default=100)   # new
    p.add_argument("--echo_every_steps", type=int, default=1_500) # new
    return p

def main(args):
    parser = build_parser()
    args = parser.parse_args()

    # --- Env
    env = SnakeEnv(Rules(SnakeCfg(grid_w=args.grid_w, grid_h=args.grid_h, relative_actions=True, seed=args.seed)))
    obs_shape = env.observation_shape()
    n_actions = env.action_space_n()

    # --- Networks
    q_net = TorchMLPQNet(obs_shape, n_actions, dueling=args.dueling, device=args.device)
    target_net = TorchMLPQNet(obs_shape, n_actions, dueling=args.dueling, device=args.device)

    # --- Replay & epsilon
    replay = RingBuffer(capacity=args.cap, seed=args.seed)
    eps = LinearDecayEpsilon(1.0, 0.05, 250_000)

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
        if stats["updates"] % 10 != 0:
            return
        logger.log(stats["step"], {
            "train/loss": stats.get("loss"),
            "train/epsilon": stats.get("epsilon"),
            "train/replay_size": stats.get("replay_size"),
            "train/q_mean": stats.get("q_mean"),
            "train/q_max": stats.get("q_max"),
            "train/q_min": stats.get("q_min"),
            "train/updates": stats.get("updates"),
            "train/grad_norm": stats.get("grad_norm"),
        })

        print(f"[step {stats['step']:>7}] upd={stats['updates']:>6} "
        f"loss={stats['loss']:.4f} eps={stats['epsilon']:.3f} "
        f"replay={stats['replay_size']:>6} qμ={stats['q_mean']:.3f} qmax={stats['q_max']:.3f}")

    def on_episode_end(ep, s):
        step = agent.state.global_step
        er, el = s["reward"], s["steps"]
        r_ema = ema_reward.update(er); l_ema = ema_length.update(el)
        win_reward.add(er); win_length.add(el)
        wr, wl = win_reward.summary(), win_length.summary()
        logger.log(step, {
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
    )

    trainer.train(num_episodes=args.episodes)
    logger.close()


if __name__ == "__main__":
    main()