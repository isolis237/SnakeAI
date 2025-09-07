# snake/rl/torch_net.py
from __future__ import annotations
from typing import Tuple, Iterable
import numpy as np
import torch
import torch.nn as nn

from .networks import QNetwork  # protocol (structural typing, no inheritance needed)
from .utils import resolve_device

class TorchMLPQNet(nn.Module):  # <-- inherit from nn.Module
    """
    PyTorch MLP Q-network that conforms to the QNetwork protocol:
      - obs_shape: (*any*), will be flattened
      - num_actions: int
      - q_values(np.ndarray) -> np.ndarray of shape (B, A)
      - hard_update_from / soft_update_from / save_weights / load_weights
    """
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        num_actions: int,
        hidden_sizes: Iterable[int] = (256, 128, 64),
        dueling: bool = False,
        device: str = resolve_device("auto"),
    ) -> None:
        super().__init__()
        self._obs_shape = tuple(obs_shape)
        self._num_actions = int(num_actions)
        self._dueling = bool(dueling)

        flat_dim = int(np.prod(self._obs_shape))

        # torso
        layers: list[nn.Module] = []
        in_dim = flat_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h

        if dueling:
            self.torso = nn.Sequential(*layers) if layers else nn.Identity()
            self.adv_head = nn.Linear(in_dim, self._num_actions)
            self.val_head = nn.Linear(in_dim, 1)
            self.net = None  # not used in dueling
        else:
            self.net = nn.Sequential(*(layers + [nn.Linear(in_dim, self._num_actions)]))
            self.torso = None
            self.adv_head = None
            self.val_head = None

        self._device = torch.device(device)
        self.to(self._device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, H, W, C) or (B, D) float tensor on self._device
        returns: (B, A) float tensor
        """
        x = obs.to(self._device, dtype=torch.float32).view(obs.size(0), -1)
        if self._dueling:
            z = self.torso(x) if self.torso is not None else x
            adv = self.adv_head(z)
            val = self.val_head(z)
            return val + adv - adv.mean(dim=1, keepdim=True)
        else:
            return self.net(x)  # type: ignore[arg-type]

    # ----- Protocol fields -----
    def obs_shape(self) -> Tuple[int, ...]:
        return self._obs_shape

    def num_actions(self) -> int:
        return self._num_actions

    @torch.no_grad()
    def q_values(self, obs_batch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(obs_batch).to(self._device, dtype=torch.float32)
        q = self.forward(x)
        return q.detach().cpu().numpy()

    # ----- Target updates -----
    def hard_update_from(self, other: "TorchMLPQNet") -> None:
        self.load_state_dict(other.state_dict())

    @torch.no_grad()
    def soft_update_from(self, other: "TorchMLPQNet", tau: float) -> None:
        for p_t, p in zip(self.parameters(), other.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    # ----- IO -----
    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self._device))
