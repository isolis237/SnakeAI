# rl/torch_cnn_qnet.py
from __future__ import annotations
from typing import Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _fan_in_uniform_(tensor, fanin=None):
    if fanin is None:
        fanin = tensor.size(0)
    bound = 1.0 / (fanin ** 0.5)
    return nn.init.uniform_(tensor, -bound, bound)


class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class TorchCNNQNet(nn.Module):
    """
    CNN-based DQN head that satisfies your QNetwork protocol:
      - obs: (N, H, W, C) float32 in [0,1]
      - outputs: (N, num_actions) Q-values
    Supports dueling arch via Value/Adv streams.
    """
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        num_actions: int,
        dueling: bool = True,
        device: str = "cuda",
        hidden: int = 256,
    ):
        super().__init__()
        assert len(obs_shape) == 3, f"Expected (H,W,C), got {obs_shape}"
        self._obs_shape = obs_shape
        self._num_actions = num_actions
        self._dueling = dueling
        self._device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        H, W, C = obs_shape
        in_ch = C

        # --- Feature extractor (small and fast, no BatchNorm for RL stability) ---
        # If you scale grid up later, you can add stride=2 to conv2 to reduce spatial dims.
        self.features = nn.Sequential(
            # (N, C, H, W) expected → we will permute in forward()
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # compute conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, H, W)
            feat = self.features(dummy)
            conv_out_dim = int(np.prod(feat.shape[1:]))

        # --- Heads ---
        if dueling:
            self.value_head = nn.Sequential(
                _Flatten(),
                nn.Linear(conv_out_dim, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )
            self.adv_head = nn.Sequential(
                _Flatten(),
                nn.Linear(conv_out_dim, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, num_actions),
            )
        else:
            self.head = nn.Sequential(
                _Flatten(),
                nn.Linear(conv_out_dim, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, num_actions),
            )

        # Initialize a bit conservatively
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _fan_in_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.to(self._device)

    # --- Protocol helpers ---
    def obs_shape(self) -> Tuple[int, ...]:
        return self._obs_shape

    def num_actions(self) -> int:
        return self._num_actions

    def hard_update_from(self, other: "TorchCNNQNet") -> None:
        self.load_state_dict(other.state_dict())

    @torch.no_grad()
    def soft_update_from(self, other: "TorchCNNQNet", tau: float) -> None:
        for p_t, p_s in zip(self.parameters(), other.parameters()):
            p_t.data.mul_(1.0 - tau).add_(tau * p_s.data)

    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str) -> None:
        sd = torch.load(path, map_location=self._device)
        self.load_state_dict(sd)

    # --- Forward / API ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, H, W, C) or (N, C, H, W). Convert to (N, C, H, W).
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.shape}")
        if x.shape[1] != self._obs_shape[2]:  # not channels-first
            # assume (N, H, W, C)
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.features(x)
        if self._dueling:
            V = self.value_head(x)               # (N, 1)
            A = self.adv_head(x)                 # (N, A)
            A = A - A.mean(dim=1, keepdim=True)  # mean-subtract for identifiability
            Q = V + A
            return Q
        else:
            return self.head(x)

    @torch.no_grad()
    def q_values(self, obs_batch: np.ndarray) -> np.ndarray:
        # obs_batch: (N, H, W, C) float32 in [0,1]
        self.eval()
        x = torch.from_numpy(obs_batch).to(self._device, dtype=torch.float32)
        q = self.forward(x)
        return q.detach().cpu().numpy()
