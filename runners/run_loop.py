# snake/runners/run_loop.py
import time, numpy as np, torch
from core.snake_rules import Rules, Config
from core.snake_env import SnakeEnv
from interfaces import StepResult
from viz.renderer_pygame import PygameRenderer
from policy.policy import TorchPolicy
from config import AppConfig

class TinyMLP(torch.nn.Module):
    def __init__(self, inp, out=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(inp, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, out)
        )
    def forward(self, x): return self.net(x)

def main(cfg: AppConfig):
    env = SnakeEnv(Rules(Config(
        grid_w=cfg.env.grid_w,
        grid_h=cfg.env.grid_h,
        relative_actions=True
    )), obs_mode="flat")
    flat_dim = np.prod(env.observation_shape())
    policy = TorchPolicy(TinyMLP(flat_dim), device=torch.device(cfg.dqn.device), obs_mode="flat")

    renderer = PygameRenderer() if cfg.render.render else None
    if renderer:
        renderer.open(cfg.env.grid_w, cfg.env.grid_h, cfg.render)

    for ep in range(cfg.runner.episodes):
        obs = env.reset(seed=ep)
        done, ret = False, 0.0
        while not done:
            a = policy.act(obs)
            step: StepResult = env.step(a)
            if renderer:
                renderer.draw(step.info["snapshot"])
                renderer.tick(cfg.runner.fps)
            obs, ret = step.obs, ret + step.reward
            done = step.terminated or step.truncated
        print(f"[ep {ep}] return={ret:.2f} score={step.info.get('score')} reason={step.info.get('reason')}")
    if renderer: renderer.close()

if __name__ == "__main__":
    # A default AppConfig would be needed to run this file directly.
    pass
