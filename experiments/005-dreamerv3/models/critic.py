"""Value critic with symlog output and EMA slow-target critic.

The critic predicts state values in symlog space, which compresses large
reward scales and stabilises training (DreamerV3 convention).

``SlowCritic`` wraps a primary critic with an EMA-updated target network
used to bootstrap TD targets without introducing moving-target instability.
"""

import copy
import torch
import torch.nn as nn
from typing import List


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    out_dim: int,
) -> nn.Sequential:
    """Build an MLP with ELU activations and LayerNorm after each hidden layer.

    Args:
        in_dim: Input dimensionality.
        hidden_dim: Width of each hidden layer.
        num_layers: Number of hidden layers.
        out_dim: Output dimensionality.

    Returns:
        nn.Sequential module.
    """
    layers: List[nn.Module] = []
    prev_dim = in_dim
    for _ in range(num_layers):
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(inplace=True),
        ])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, out_dim))
    return nn.Sequential(*layers)


class Critic(nn.Module):
    """MLP value estimator outputting symlog-scaled scalar values.

    The final linear layer produces a single scalar per batch element. The
    output is interpreted as a symlog-transformed value; callers should apply
    ``symexp`` to recover the original scale if needed.

    Args:
        latent_dim: Dimensionality of the input latent state vector.
        hidden_dim: Width of MLP hidden layers.
        num_layers: Number of hidden layers.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.net = _build_mlp(latent_dim, hidden_dim, num_layers, out_dim=1)

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Predict scalar value for a batch of latent states.

        Args:
            latent_state: (B, latent_dim) latent state tensor.

        Returns:
            (B,) predicted value in symlog space.
        """
        return self.net(latent_state).squeeze(-1)


class SlowCritic:
    """EMA target critic for stable value bootstrapping.

    Maintains a frozen copy of ``critic`` that is updated towards the live
    critic via exponential moving average. All forward passes through the
    target are done with ``torch.no_grad()``.

    Args:
        critic: Primary (trained) Critic module.
        decay: EMA decay factor. Higher = slower target update (default 0.98).
    """

    def __init__(self, critic: Critic, decay: float = 0.98):
        self.critic = critic
        self.target = copy.deepcopy(critic)
        self.decay = decay

    def update(self) -> None:
        """Soft-update target network parameters toward the live critic.

        Uses the lerp rule:  target = decay * target + (1 - decay) * critic
        """
        for param, target_param in zip(
            self.critic.parameters(), self.target.parameters()
        ):
            target_param.data.lerp_(param.data, 1.0 - self.decay)

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Predict scalar value using the target (slow) network.

        Args:
            latent_state: (B, latent_dim) latent state tensor.

        Returns:
            (B,) predicted value in symlog space, computed with no gradients.
        """
        with torch.no_grad():
            return self.target(latent_state)
