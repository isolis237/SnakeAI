# main.py
import argparse

from runners.run_snake import main as snake
from runners.run_agent import main as snake_agent
from runners.run_model import main as model
from runners.run_cnn import main as cnn
from runners.run_astar import main as astar
from runners.compare_models import main as compare_models

from rl.utils import resolve_device

def run_snake(): snake()
def run_snake_agent(): snake_agent()
def run_model(): model()
def run_cnn(): cnn()
def run_astar(): astar()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["snake", "snake_agent", "model", "astar", "compare", "cnn"])
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == "snake":
        run_snake()
    elif args.mode == "snake_agent":
        run_snake_agent()
    elif args.mode == "cnn":
        run_cnn()
    elif args.mode == "astar":
        run_astar()
    elif args.mode == "model":
        model()
    elif args.mode == "compare":  
        compare_models()

if __name__ == "__main__":
    main()
