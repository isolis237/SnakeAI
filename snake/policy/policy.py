# snake/policy/policy.py
from __future__ import annotations
from typing import Protocol
import numpy as np
import torch
import torch.nn as nn

class Policy(Protocol):
    def act(self, obs: np.ndarray) -> int: ...
    def act_batch(self, obs_batch: np.ndarray) -> np.ndarray: ...

class TorchPolicy:
    def __init__(self, model: nn.Module, device: torch.device, obs_mode: str = "flat"):
        self.model = model.to(device).eval()
        self.device = device
        self.obs_mode = obs_mode

    def act(self, obs: np.ndarray) -> int:
        x = torch.from_numpy(obs).float().to(self.device)
        if x.ndim == 1: x = x.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
        return int(torch.argmax(logits, dim=1).item())

    def act_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        xb = torch.from_numpy(obs_batch).float().to(self.device)
        with torch.no_grad():
            logits = self.model(xb)
        return torch.argmax(logits, dim=1).cpu().numpy()
