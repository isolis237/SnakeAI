# policy.py
from __future__ import annotations
from typing import Protocol, Callable, Optional
import numpy as np
import torch
import torch.nn as nn
import random

class Policy(Protocol):
    def act(self, obs: np.ndarray) -> int: ...
    def act_batch(self, obs_batch: np.ndarray) -> np.ndarray: ...

class TorchPolicy:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        obs_mode: str = "flat",
        vis_callback: Optional[Callable[..., None]] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.obs_mode = obs_mode
        self.vis_callback = vis_callback
        self._acts = {}  # filled by activation hooks

    def act(self, obs: np.ndarray) -> int:
        self.model.eval()

        x = torch.from_numpy(obs).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device, non_blocking=True)
        with torch.no_grad():
            logits = self.model(x)
        if self.vis_callback is not None:
            self.vis_callback(activations=self._acts,
                            logits=logits.detach().cpu().numpy()[0])
        return int(torch.argmax(logits, dim=1).item())

    def act_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        xb = torch.from_numpy(obs_batch).float().to(self.device, non_blocking=True)
        with torch.no_grad():
            logits = self.model(xb)
        return torch.argmax(logits, dim=1).cpu().numpy()
