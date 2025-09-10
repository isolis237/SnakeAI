# main.py
import argparse

from runners.run_snake import main as snake
from runners.run_agent import main as snake_agent
from runners.run_model import main as model_playback   # <--- NEW
from rl.utils import resolve_device

def run_snake(args): snake(args)
def run_snake_agent(args): snake_agent(args)
def run_model(args): model_playback(args)               # <--- NEW

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["snake", "snake_agent", "model", "loop", "multi", "nnview"])
    # --- shared/game args ---
    p.add_argument("--grid-w", type=int, default=12)
    p.add_argument("--grid-h", type=int, default=12)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=resolve_device("auto"))
    p.add_argument("--dueling", action="store_true")
    p.add_argument("--max_ep_steps", type=int, default=0)

    # --- training-specific ---
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=8_000)
    p.add_argument("--cap", type=int, default=150_000)

    # --- live viewer (both train & playback) ---
    p.add_argument("--live_view", action="store_true")
    p.add_argument("--view_fps", type=int, default=12)
    p.add_argument("--record_dir", type=str, default=None)
    p.add_argument("--cell_px", type=int, default=24)
    p.add_argument("--grid_lines", action="store_true")
    p.add_argument("--hide_hud", action="store_true")
    p.add_argument("--stream_ep_prob", type=float, default=.010)
    p.add_argument("--best_metric", choices=["reward","final_score"], default="reward")
    p.add_argument("--viewer_seed", type=int, default=0)
    p.add_argument("--player_buffer", type=int, default=4)
    p.add_argument("--player_gap", type=float, default=0.35)

    # --- playback-specific ---
    p.add_argument("--ckpt_dir", type=str, default="runs/snake_dqn")
    p.add_argument("--tag", type=str, default="dqn_best")
    p.add_argument("--play_episodes", type=int, default=3)

    # (legacy UI args you had like fps, echo_every_steps, etc. – keep if you still use them)
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == "snake":
        run_snake(args)
    elif args.mode == "snake_agent":
        run_snake_agent(args)
    elif args.mode == "model":                 # <--- NEW
        run_model(args)
    elif args.mode == "loop":
        run_loop(args)
    elif args.mode == "multi":
        run_multi(args)
    elif args.mode == "nnview":
        if args.flat_dim is None:
            args.flat_dim = args.grid_w * args.grid_h * 3
        run_nn_viewer(args)

if __name__ == "__main__":
    main()
