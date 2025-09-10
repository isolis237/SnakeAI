# main.py
import argparse

from runners.run_snake import main as snake
from runners.run_agent import main as snake_agent

from rl.utils import resolve_device

def run_snake(): snake()
def run_snake_agent(): snake_agent()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["snake", "snake_agent", "model", "loop", "multi", "nnview"])
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == "snake":
        run_snake()
    elif args.mode == "snake_agent":
        run_snake_agent()

if __name__ == "__main__":
    main()
