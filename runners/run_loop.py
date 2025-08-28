# snake/runners/run_loop.py
import time, numpy as np, torch
from core.snake_rules import Rules, Config
from core.snake_env import SnakeEnv
from core.interfaces import StepResult
from viz.renderer import Renderer
from policy.policy import TorchPolicy

class TinyMLP(torch.nn.Module):
    def __init__(self, inp, out=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(inp, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, out)
        )
    def forward(self, x): return self.net(x)

def main(render=True, episodes=5, fps=12, obs_mode="flat"):
    cfg = Config(grid_w=12, grid_h=12, relative_actions=True)
    env = SnakeEnv(Rules(cfg), obs_mode=obs_mode)
    flat_dim = np.prod(env.observation_shape())
    policy = TorchPolicy(TinyMLP(flat_dim), device=torch.device("cpu"), obs_mode=obs_mode)

    renderer = Renderer(cell_px=24) if render else None
    if renderer: renderer.create_window(cfg.grid_w, cfg.grid_h, "Snake — Eval")

    for ep in range(episodes):
        obs = env.reset(seed=ep)
        done, ret = False, 0.0
        while not done:
            a = policy.act(obs)
            step: StepResult = env.step(a)
            if renderer:
                renderer.draw(step.info["snapshot"])
                renderer.tick(fps)
            obs, ret = step.obs, ret + step.reward
            done = step.terminated or step.truncated
        print(f"[ep {ep}] return={ret:.2f} score={step.info.get('score')} reason={step.info.get('reason')}")
    if renderer: renderer.close()

if __name__ == "__main__":
    main()
