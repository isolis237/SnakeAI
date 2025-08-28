# snake/runners/run_policy.py
import numpy as np
import torch
from snake.policy.policy import TorchPolicy

class TinyMLP(torch.nn.Module):
    def __init__(self, inp, out=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(inp, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, out)
        )
    def forward(self, x): return self.net(x)

def main(flat_dim=12*12*3, iters=1000, seed=0):
    rng = np.random.default_rng(seed)
    model = TinyMLP(flat_dim)
    pol = TorchPolicy(model, device=torch.device("cpu"), obs_mode="flat")

    # synthetic “obs stream” (could also load from .npz replay)
    cnt = np.zeros(3, dtype=int)
    for _ in range(iters):
        obs = rng.random(flat_dim, dtype=np.float32)
        a = pol.act(obs)
        cnt[a] += 1
    print("action histogram:", cnt)

if __name__ == "__main__":
    main()
