# main.py
import argparse
from config import AppConfig, EnvConfig, RenderConfig, DQNConfig, RunnerConfig
from runners.run_snake import main as run_snake
from runners.run_agent import main as run_snake_agent
from runners.run_loop import main as run_loop
from runners.run_multi import main as run_multi
# from runners.run_nn_viewer import main as run_nn_viewer # Assuming this exists
from rl.utils import resolve_device

def parse_args() -> AppConfig:
    p = argparse.ArgumentParser()
    # top-level
    p.add_argument("mode", choices=["snake", "snake_agent", "loop", "multi", "nnview"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    # env
    p.add_argument("--grid-w", type=int, default=12)
    p.add_argument("--grid-h", type=int, default=12)
    p.add_argument("--n-actions", type=int, default=3)
    p.add_argument("--max_ep_steps", type=int, default=0)

    # render
    p.add_argument("--render", action="store_true")
    p.add_argument("--live_view", action="store_true")
    p.add_argument("--view_fps", type=int, default=12)
    p.add_argument("--record_dir", type=str, default=None)
    p.add_argument("--cell_px", type=int, default=24)
    p.add_argument("--grid_lines", action="store_true")
    p.add_argument("--hide_hud", action="store_true")

    # dqn
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--warmup", type=int, default=8_000)
    p.add_argument("--cap", type=int, default=150_000)
    p.add_argument("--dueling", action="store_true")

    # runner
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--episodes-per-env", type=int, default=3)
    p.add_argument("--log_every_updates", type=int, default=100)   
    p.add_argument("--echo_every_steps", type=int, default=1_500)
    p.add_argument("--view_every_updates", type=int, default=500)
    p.add_argument("--stream_ep_prob", type=float, default=.010)
    p.add_argument("--best_metric", choices=["reward","final_score"], default="reward")
    p.add_argument("--viewer_seed", type=int, default=0)
    p.add_argument("--player_buffer", type=int, default=4)
    p.add_argument("--player_gap", type=float, default=0.35)
    p.add_argument("--flat-dim", type=int, default=None)

    args = p.parse_args()

    env_cfg = EnvConfig(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        n_actions=args.n_actions,
        max_ep_steps=args.max_ep_steps,
    )

    render_cfg = RenderConfig(
        render=args.render,
        live_view=args.live_view,
        view_fps=args.view_fps,
        record_dir=args.record_dir,
        cell_px=args.cell_px,
        grid_lines=args.grid_lines,
        show_hud=not args.hide_hud,
    )

    dqn_cfg = DQNConfig(
        lr=args.lr,
        batch_size=args.batch,
        min_replay_before_learn=args.warmup,
        replay_capacity=args.cap,
        dueling=args.dueling,
        device=resolve_device(args.device),
        seed=args.seed,
    )

    runner_cfg = RunnerConfig(
        mode=args.mode,
        episodes=args.episodes,
        fps=args.fps,
        n_envs=args.n_envs,
        episodes_per_env=args.episodes_per_env,
        log_every_updates=args.log_every_updates,
        echo_every_steps=args.echo_every_steps,
        view_every_updates=args.view_every_updates,
        stream_ep_prob=args.stream_ep_prob,
        best_metric=args.best_metric,
        viewer_seed=args.viewer_seed,
        player_buffer=args.player_buffer,
        player_gap=args.player_gap,
        flat_dim=args.flat_dim,
    )

    # The mode is also in runner_cfg, but we need it here to call the right runner.
    # We'll pass the full config to the runner.
    cfg = AppConfig(
        env=env_cfg,
        render=render_cfg,
        dqn=dqn_cfg,
        runner=runner_cfg,
    )
    cfg.runner.mode = args.mode

    return cfg


def main():
    cfg = parse_args()
    if cfg.runner.mode == "snake":
        run_snake(cfg)
    elif cfg.runner.mode == "snake_agent":
        run_snake_agent(cfg)
    elif cfg.runner.mode == "loop":
        run_loop(cfg)
    elif cfg.runner.mode == "multi":
        run_multi(cfg)
    elif cfg.runner.mode == "nnview":
        # default flat_dim if not provided
        if cfg.runner.flat_dim is None:
            cfg.runner.flat_dim = cfg.env.grid_w * cfg.env.grid_h * 3
        # run_nn_viewer(cfg) # I don't have this runner, so I will comment it out.


if __name__ == "__main__":
    main()
