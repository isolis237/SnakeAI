# main.py
import argparse
import numpy as np
import torch

# ---- snake side ----
from snake.core.snake_rules import Rules, Config
from snake.core.snake_env import SnakeEnv
from snake.viz.renderer import Renderer
from snake.viz.keyboard import Keyboard

# ---- policy side ----
from snake.policy.policy import TorchPolicy


# simple example model (replace with your real one)
class TinyMLP(torch.nn.Module):
    def __init__(self, inp, out=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(inp, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, out)
        )
    def forward(self, x): return self.net(x)


def run_snake(args):
    """Run Snake game only (human control, no NN)."""
    env = SnakeEnv(Rules(Config(grid_w=args.grid_w, grid_h=args.grid_h)))
    rend = Renderer(cell_px=args.cell_px)
    rend.create_window(args.grid_w, args.grid_h, "Snake — Human")
    kbd = Keyboard(relative=True)

    obs = env.reset()
    done = False
    while not done:
        key = kbd.poll()
        if key == "quit": break
        action = key if isinstance(key, int) else 0
        step = env.step(action)
        rend.draw(step.info["snapshot"])
        rend.tick(args.fps)
        done = step.terminated or step.truncated
    rend.close()


def run_policy(args):
    """Run NN only (feed it random observations, see outputs)."""
    flat_dim = args.grid_w * args.grid_h * 3
    model = TinyMLP(flat_dim)
    pol = TorchPolicy(model, device=torch.device("cpu"), obs_mode="flat")

    rng = np.random.default_rng(args.seed)
    cnt = np.zeros(3, dtype=int)
    for _ in range(100):
        obs = rng.random(flat_dim, dtype=np.float32)
        a = pol.act(obs)
        cnt[a] += 1
    print("NN-only mode action histogram:", cnt)


def run_loop(args):
    """Run NN + Snake together (training/eval loop)."""
    cfg = Config(grid_w=args.grid_w, grid_h=args.grid_h)
    env = SnakeEnv(Rules(cfg), obs_mode="flat")
    flat_dim = np.prod(env.observation_shape())
    policy = TorchPolicy(TinyMLP(flat_dim), device=torch.device("cpu"), obs_mode="flat")

    renderer = Renderer(cell_px=args.cell_px) if args.render else None
    if renderer:
        renderer.create_window(cfg.grid_w, cfg.grid_h, "Snake — Eval")

    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        done, ret = False, 0.0
        while not done:
            a = policy.act(obs)
            step = env.step(a)
            if renderer:
                renderer.draw(step.info["snapshot"])
                renderer.tick(args.fps)
            obs, ret = step.obs, ret + step.reward
            done = step.terminated or step.truncated
        print(f"[ep {ep}] return={ret:.2f} score={step.info.get('score')}")
    if renderer:
        renderer.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["snake", "policy", "loop"])
    p.add_argument("--grid-w", type=int, default=12)
    p.add_argument("--grid-h", type=int, default=12)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--cell-px", type=int, default=24)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--episodes-per-env", type=int, default=3)
    p.add_argument("--render", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "snake":
        run_snake(args)
    elif args.mode == "policy":
        run_policy(args)
    elif args.mode == "loop":
        run_loop(args)
    elif args.mode == "multi":
        run_multi(args)



if __name__ == "__main__":
    main()
