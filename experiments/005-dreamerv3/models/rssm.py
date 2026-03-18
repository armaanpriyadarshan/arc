"""Recurrent State-Space Model (RSSM) — core of DreamerV3.

The RSSM factorises the world model into:
- A deterministic recurrent state h_t (GRU hidden).
- A stochastic categorical state z_t sampled from a learned prior or posterior.

During training (observe_step) both a prior p(z|h) and a posterior q(z|h,x)
are computed; the KL divergence between them is used as a training signal.

During imagination (imagine_step) only the prior is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RSSM(nn.Module):
    """Recurrent State-Space Model.

    Maintains latent state composed of:
    - Deterministic state h_t: GRU hidden vector of size ``gru_hidden_dim``.
    - Stochastic state z_t: straight-through categorical tensor of shape
      (B, num_categories, num_classes), flattened to (B, stochastic_dim)
      when concatenated with h.

    Three sub-networks:
    1. Sequence model  : h_t = GRU(h_{t-1}, [z_{t-1}, action_embed(a_{t-1})])
    2. Prior           : p(z_t | h_t)          — used during imagination
    3. Posterior       : q(z_t | h_t, embed_t) — used during observation

    Args:
        embed_dim: Dimension of encoded observation (output of Encoder).
        action_dim: Number of discrete action types (one-hot encoded).
        gru_hidden_dim: Size of the GRU hidden state (deterministic component).
        num_categories: Number of independent categorical variables in z.
        num_classes: Number of classes per categorical variable.
        hidden_dim: Width of prior/posterior MLP hidden layers.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        action_dim: int = 7,
        gru_hidden_dim: int = 256,
        num_categories: int = 8,
        num_classes: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        stoch_dim = num_categories * num_classes

        # Project discrete action index (one-hot) to hidden_dim
        self.action_embed = nn.Linear(action_dim, hidden_dim)

        # GRU: input = [prev_z (stoch_dim), action_embed (hidden_dim)]
        self.gru_cell = nn.GRUCell(
            input_size=stoch_dim + hidden_dim,
            hidden_size=gru_hidden_dim,
        )

        # Prior p(z | h): predicts stochastic state from deterministic state alone
        self.prior_mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, stoch_dim),
        )

        # Posterior q(z | h, embed): conditions on observed embedding too
        self.posterior_mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, stoch_dim),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initial_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialised (h, z) tuple.

        Args:
            batch_size: Number of parallel environments / sequences.
            device: Target device.

        Returns:
            Tuple of (h, z) where:
                h: (B, gru_hidden_dim) zeros
                z: (B, num_categories, num_classes) zeros
        """
        h = torch.zeros(batch_size, self.gru_hidden_dim, device=device)
        z = torch.zeros(
            batch_size, self.num_categories, self.num_classes, device=device
        )
        return h, z

    def observe_step(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        prev_action: torch.Tensor,
        embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one step with an observed embedding.

        Steps:
        1. Embed the action and concatenate with flattened prev_z.
        2. Advance GRU to get new deterministic state h.
        3. Compute prior logits p(z|h).
        4. Compute posterior logits q(z|h, embed).
        5. Sample z from the posterior (straight-through).

        Args:
            prev_h: (B, gru_hidden_dim) previous deterministic state.
            prev_z: (B, num_categories, num_classes) previous stochastic state.
            prev_action: (B,) int64 action indices.
            embed: (B, embed_dim) encoded observation.

        Returns:
            Tuple of:
                h               : (B, gru_hidden_dim) new deterministic state
                z_posterior     : (B, num_categories, num_classes) sampled posterior
                prior_logits    : (B, stochastic_dim) raw prior logits
                posterior_logits: (B, stochastic_dim) raw posterior logits
        """
        h = self._advance_gru(prev_h, prev_z, prev_action)
        prior_logits = self.prior_mlp(h)
        posterior_logits = self.posterior_mlp(torch.cat([h, embed], dim=-1))
        z_posterior = self._sample_stochastic(posterior_logits)
        return h, z_posterior, prior_logits, posterior_logits

    def imagine_step(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one step without an observation (imagination / dreaming).

        Uses only the prior — no real embedding is available.

        Args:
            prev_h: (B, gru_hidden_dim) previous deterministic state.
            prev_z: (B, num_categories, num_classes) previous stochastic state.
            prev_action: (B,) int64 action indices.

        Returns:
            Tuple of:
                h           : (B, gru_hidden_dim) new deterministic state
                z_prior     : (B, num_categories, num_classes) sampled prior
                prior_logits: (B, stochastic_dim) raw prior logits
        """
        h = self._advance_gru(prev_h, prev_z, prev_action)
        prior_logits = self.prior_mlp(h)
        z_prior = self._sample_stochastic(prior_logits)
        return h, z_prior, prior_logits

    def get_latent(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Concatenate deterministic and flattened stochastic state.

        Args:
            h: (B, gru_hidden_dim) deterministic state.
            z: (B, num_categories, num_classes) stochastic state.

        Returns:
            (B, latent_dim) combined latent vector.
        """
        return torch.cat([h, z.flatten(start_dim=1)], dim=-1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stochastic_dim(self) -> int:
        """Flattened size of the stochastic state."""
        return self.num_categories * self.num_classes

    @property
    def latent_dim(self) -> int:
        """Total latent dimension = gru_hidden_dim + stochastic_dim."""
        return self.gru_hidden_dim + self.stochastic_dim

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _advance_gru(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one GRU step.

        Args:
            prev_h: (B, gru_hidden_dim)
            prev_z: (B, num_categories, num_classes)
            prev_action: (B,) int64 action indices.

        Returns:
            h: (B, gru_hidden_dim) new hidden state.
        """
        # One-hot encode action
        action_one_hot = F.one_hot(
            prev_action.long(), num_classes=self.action_dim
        ).float()                                        # (B, action_dim)
        action_emb = self.action_embed(action_one_hot)  # (B, hidden_dim)

        z_flat = prev_z.flatten(start_dim=1)            # (B, stochastic_dim)
        gru_input = torch.cat([z_flat, action_emb], dim=-1)  # (B, stochastic_dim + hidden_dim)
        return self.gru_cell(gru_input, prev_h)         # (B, gru_hidden_dim)

    def _sample_stochastic(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical distribution with straight-through gradients.

        Reshapes logits to (B, num_categories, num_classes), applies softmax
        per category, then returns a one-hot sample that passes gradients
        through the soft probabilities (straight-through estimator).

        Args:
            logits: (B, num_categories * num_classes) raw logits.

        Returns:
            z: (B, num_categories, num_classes) straight-through sample.
        """
        B = logits.shape[0]
        logits_2d = logits.view(B, self.num_categories, self.num_classes)
        probs = F.softmax(logits_2d, dim=-1)  # (B, num_categories, num_classes)

        # Sample hard one-hot indices
        indices = torch.distributions.Categorical(probs=probs).sample()  # (B, num_categories)
        one_hot = F.one_hot(indices, num_classes=self.num_classes).float()  # (B, num_cat, num_cls)

        # Straight-through: use one_hot in forward, probs in backward
        z = one_hot - probs.detach() + probs
        return z
