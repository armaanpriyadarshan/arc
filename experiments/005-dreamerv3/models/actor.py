"""Hierarchical actor that handles both simple and click (ACTION6) actions.

The actor maps latent states to action distributions. ACTION6 is special: it
requires x,y grid coordinates. The hierarchical design first samples an action
type, then conditionally samples spatial coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


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


# Index of the click action within the action type space.
# Set to -1 to disable click (e.g., LockSmith only uses ACTION1-4).
CLICK_ACTION_IDX = -1


class HierarchicalActor(nn.Module):
    """Hierarchical actor with click (ACTION6) support.

    A shared MLP trunk processes the latent state, then three independent
    heads produce logits:
    - action_type_head  : 7 logits over base actions
    - click_x_head      : 64 logits over grid x-coordinate
    - click_y_head      : 64 logits over grid y-coordinate

    During sampling, x and y are only used when action_type == CLICK_ACTION_IDX
    (ACTION6). Log-probability and entropy computations reflect this hierarchy.

    Args:
        latent_dim: Dimensionality of the input latent state vector.
        num_actions: Number of discrete action types (default 7).
        grid_size: Grid side length for click coordinates (default 64).
        hidden_dim: Width of MLP hidden layers.
        num_layers: Number of hidden layers in shared trunk.
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int = 7,
        grid_size: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.grid_size = grid_size

        self.trunk = _build_mlp(latent_dim, hidden_dim, num_layers, hidden_dim)
        self.action_type_head = nn.Linear(hidden_dim, num_actions)
        self.click_x_head = nn.Linear(hidden_dim, grid_size)
        self.click_y_head = nn.Linear(hidden_dim, grid_size)

    def forward(
        self, latent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action logits from a latent state batch.

        Args:
            latent_state: (B, latent_dim) latent state tensor.

        Returns:
            Tuple of:
                action_type_logits : (B, num_actions)
                click_x_logits     : (B, grid_size)
                click_y_logits     : (B, grid_size)
        """
        features = self.trunk(latent_state)
        return (
            self.action_type_head(features),
            self.click_x_head(features),
            self.click_y_head(features),
        )

    def sample(
        self, latent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy.

        Samples action type first. If action_type == CLICK_ACTION_IDX, also
        samples x and y coordinates; otherwise x and y are set to zero.

        Log-probability is:
        - log p(type)               for non-click actions
        - log p(type) + log p(x) + log p(y)  for click actions

        Entropy is the sum of entropies for all three distributions (always).

        Args:
            latent_state: (B, latent_dim) latent state tensor.

        Returns:
            Tuple of:
                action_type : (B,) int64 sampled action type indices
                x           : (B,) int64 sampled x-coordinates (0 if non-click)
                y           : (B,) int64 sampled y-coordinates (0 if non-click)
                log_prob    : (B,) log probability of the sampled action
                entropy     : (B,) total entropy of the action distribution
        """
        type_logits, x_logits, y_logits = self.forward(latent_state)

        type_dist = torch.distributions.Categorical(logits=type_logits)
        x_dist = torch.distributions.Categorical(logits=x_logits)
        y_dist = torch.distributions.Categorical(logits=y_logits)

        action_type = type_dist.sample()   # (B,)
        x = x_dist.sample()               # (B,)
        y = y_dist.sample()               # (B,)

        is_click = (action_type == CLICK_ACTION_IDX).float()  # (B,)

        type_lp = type_dist.log_prob(action_type)   # (B,)
        x_lp = x_dist.log_prob(x)                   # (B,)
        y_lp = y_dist.log_prob(y)                   # (B,)

        # Only add spatial log-probs for click actions
        log_prob = type_lp + is_click * (x_lp + y_lp)

        # Entropy: always sum all three distributions (full distribution entropy)
        entropy = type_dist.entropy() + x_dist.entropy() + y_dist.entropy()

        # Zero out coordinates for non-click actions
        x = (x * is_click.long()).long()
        y = (y * is_click.long()).long()

        return action_type, x, y, log_prob, entropy

    def log_prob(
        self,
        latent_state: torch.Tensor,
        action_type: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of given actions under the current policy.

        Args:
            latent_state: (B, latent_dim) latent state tensor.
            action_type: (B,) int64 action type indices.
            x: (B,) int64 x-coordinate (ignored for non-click actions).
            y: (B,) int64 y-coordinate (ignored for non-click actions).

        Returns:
            (B,) log probability of each action.
        """
        type_logits, x_logits, y_logits = self.forward(latent_state)

        type_dist = torch.distributions.Categorical(logits=type_logits)
        x_dist = torch.distributions.Categorical(logits=x_logits)
        y_dist = torch.distributions.Categorical(logits=y_logits)

        is_click = (action_type == CLICK_ACTION_IDX).float()

        type_lp = type_dist.log_prob(action_type)
        x_lp = x_dist.log_prob(x)
        y_lp = y_dist.log_prob(y)

        return type_lp + is_click * (x_lp + y_lp)
