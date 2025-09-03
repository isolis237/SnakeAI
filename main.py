# main.py
import argparse

from runners.run_snake import main as snake
from runners.run_agent import main as snake_agent

def run_snake(args):
    snake(args)

def run_snake_agent(args):
    snake_agent(args)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["snake", "snake_agent", "loop", "multi", "nnview"])
    p.add_argument("--grid-w", type=int, default=16)
    p.add_argument("--grid-h", type=int, default=16)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--cell-px", type=int, default=64)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--episodes-per-env", type=int, default=3)
    p.add_argument("--render", action="store_true")
    p.add_argument("--flat-dim", type=int, default=None, help="override input dim for NN viewer")
    p.add_argument("--n-actions", type=int, default=3, help="3=relative, 4=absolute")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "snake":
        run_snake(args)
    elif args.mode == "snake_agent":
        run_snake_agent(args)
    elif args.mode == "loop":
        run_loop(args)
    elif args.mode == "multi":
        run_multi(args)
    elif args.mode == "nnview":
        # default flat_dim if not provided
        if args.flat_dim is None:
            args.flat_dim = args.grid_w * args.grid_h * 3
        run_nn_viewer(args)



if __name__ == "__main__":
    main()
