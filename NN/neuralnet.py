import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Iterable, Literal, Optional

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_actions: int = 3,
        hidden_sizes: Iterable[int] = (16, 16),
        activation: Literal["relu", "leaky_relu", "tanh", "gelu"] = "relu",
        use_batchnorm: bool = False,
        dropout_p: float = 0.0,
        weight_init: Literal["kaiming", "xavier", "none"] = "kaiming",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_actions = int(n_actions)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.activation_name = activation
        self.use_batchnorm = bool(use_batchnorm)
        self.dropout_p = float(dropout_p)
        self.weight_init = weight_init

        layers: list[nn.Module] = []
        prev = self.input_dim

        for i, h in enumerate(self.hidden_sizes):
            layers.append(nn.Linear(prev, h))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if self.dropout_p > 0:
                layers.append(nn.Dropout(self.dropout_p))
            prev = h

        # Final layer -> action logits (no activation here)
        layers.append(nn.Linear(prev, self.n_actions))

        # Wrap as a single Sequential for cleanliness
        self.net = nn.Sequential(*layers)

        if self.weight_init != "none":
            self._init_weights(self.weight_init, nonlinearity="relu")


    def forward(self, x: torch.Tensor, return_probs: bool = False, temperature: float = 1.0) -> torch.Tensor:
        """
        x: (batch, input_dim)
        return_probs=False -> logits (preferred for nn.CrossEntropyLoss)
        return_probs=True  -> softmax probabilities that sum to 1 across actions
        """
        # Basic shape check (helpful when starting out)
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected x of shape (B, {self.input_dim}), got {tuple(x.shape)}")

        logits = self.net(x)

        if return_probs:
            if temperature <= 0:
                raise ValueError("temperature must be > 0")
            return F.softmax(logits / temperature, dim=1)

        return logits  # raw scores, one per action

    def _init_weights(self, scheme: str, nonlinearity: str = "relu") -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if scheme == "kaiming":
                    # Kaiming for ReLU-like activations
                    nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
                elif scheme == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                else:
                    pass
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
