"""World model: Encoder + RSSM + Decoder + reward head + continue head.

Implements the full DreamerV3 world model. Given sequences of real
observations and actions it produces:
- Posterior latent states (for the representation loss)
- Prior latent states (for the KL loss)
- Decoded observation logits (for the reconstruction loss)
- Predicted rewards and episode-continue probabilities

During imagination it rolls out the policy forward from a given latent state
without observing the environment, producing synthetic trajectories for the
actor-critic update.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from .encoder import Encoder
from .decoder import Decoder
from .rssm import RSSM
from .actor import HierarchicalActor


# ---------------------------------------------------------------------------
# Symlog utilities
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm: sign(x) * ln(|x| + 1).

    Compresses large-magnitude values while preserving sign and remaining
    smooth near zero.

    Args:
        x: Input tensor.

    Returns:
        symlog-transformed tensor.
    """
    return x.sign() * (x.abs() + 1.0).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1).

    Args:
        x: symlog-transformed tensor.

    Returns:
        Original-scale tensor.
    """
    return x.sign() * (x.abs().exp() - 1.0)


# ---------------------------------------------------------------------------
# Minimal MLP head
# ---------------------------------------------------------------------------

def _build_head(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Single hidden-layer MLP with ELU + LayerNorm, used for reward/continue heads.

    Args:
        in_dim: Input size.
        hidden_dim: Hidden layer size.
        out_dim: Output size.

    Returns:
        nn.Sequential head module.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ELU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


# ---------------------------------------------------------------------------
# WorldModel
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """Full DreamerV3 world model.

    Combines:
    - ``Encoder``      : (B, 16, 64, 64) → (B, embed_dim)
    - ``RSSM``         : latent state transitions
    - ``Decoder``      : (B, latent_dim) → (B, 16, 64, 64) logits
    - ``reward_head``  : (B, latent_dim) → (B,) symlog reward
    - ``continue_head``: (B, latent_dim) → (B,) continue logit

    Config dict keys used:
        encoder_channels        : list[int]  e.g. [32, 64, 128]
        embed_dim               : int        e.g. 1024
        gru_hidden_dim          : int        e.g. 256
        stochastic_categories   : int        e.g. 8
        stochastic_classes      : int        e.g. 8
        hidden_dim              : int        e.g. 256
        num_colors              : int        16
        grid_size               : int        64
        num_base_actions        : int        7

    Args:
        config: Dictionary of hyperparameters described above.
    """

    def __init__(self, config: dict):
        super().__init__()

        encoder_channels: List[int] = config.get("encoder_channels", [32, 64, 128])
        embed_dim: int = config.get("embed_dim", 1024)
        gru_hidden_dim: int = config.get("gru_hidden_dim", 256)
        num_categories: int = config.get("stochastic_categories", 8)
        num_classes: int = config.get("stochastic_classes", 8)
        hidden_dim: int = config.get("hidden_dim", 256)
        num_colors: int = config.get("num_colors", 16)
        num_actions: int = config.get("num_base_actions", 7)

        self.encoder = Encoder(
            in_channels=num_colors,
            channels=encoder_channels,
            embed_dim=embed_dim,
        )

        self.rssm = RSSM(
            embed_dim=embed_dim,
            action_dim=num_actions,
            gru_hidden_dim=gru_hidden_dim,
            num_categories=num_categories,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )

        latent_dim = self.rssm.latent_dim

        self.decoder = Decoder(
            latent_dim=latent_dim,
            channels=list(reversed(encoder_channels)),
            out_channels=num_colors,
        )

        self.reward_head = _build_head(latent_dim, hidden_dim, out_dim=1)
        self.continue_head = _build_head(latent_dim, hidden_dim, out_dim=1)

    # ------------------------------------------------------------------
    # Sequence observation
    # ------------------------------------------------------------------

    def observe(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Process a sequence of real observations through the world model.

        Runs the RSSM's observe_step over T timesteps, encoding observations
        and collecting latent states, KL terms, and decoded predictions.

        Args:
            observations: Float tensor (B, T, 16, 64, 64). One-hot or
                          normalised color values.
            actions: Int64 tensor (B, T) of action-type indices. The action
                     at time t is ``actions[:, t]`` which is the action taken
                     *after* observing frame t (used to transition to t+1).

        Returns:
            Dictionary containing:
                latents         : (B, T, latent_dim) combined latent states
                prior_logits    : (B, T, stochastic_dim) prior distribution logits
                posterior_logits: (B, T, stochastic_dim) posterior distribution logits
                decoded_obs     : (B, T, 16, 64, 64) reconstruction logits
                predicted_rewards   : (B, T) symlog reward predictions
                predicted_continues : (B, T) continue logits (pre-sigmoid)
        """
        B, T, C, H, W = observations.shape
        device = observations.device

        h, z = self.rssm.initial_state(B, device)

        # Encode all observations in one batch pass for efficiency
        obs_flat = observations.view(B * T, C, H, W)       # (B*T, 16, 64, 64)
        embeds_flat = self.encoder(obs_flat)               # (B*T, embed_dim)
        embeds = embeds_flat.view(B, T, -1)                # (B, T, embed_dim)

        latents: List[torch.Tensor] = []
        prior_logits_list: List[torch.Tensor] = []
        posterior_logits_list: List[torch.Tensor] = []

        # Initial dummy action (zeros) for the first step
        prev_action = torch.zeros(B, dtype=torch.long, device=device)

        for t in range(T):
            h, z_post, prior_l, post_l = self.rssm.observe_step(
                h, z, prev_action, embeds[:, t]
            )
            z = z_post
            latent = self.rssm.get_latent(h, z)

            latents.append(latent)
            prior_logits_list.append(prior_l)
            posterior_logits_list.append(post_l)

            prev_action = actions[:, t]

        # Stack over time dimension
        latents_t = torch.stack(latents, dim=1)                     # (B, T, latent_dim)
        prior_logits_t = torch.stack(prior_logits_list, dim=1)      # (B, T, stoch_dim)
        posterior_logits_t = torch.stack(posterior_logits_list, dim=1)  # (B, T, stoch_dim)

        # Decode all latents in one batch pass
        lat_flat = latents_t.view(B * T, -1)                        # (B*T, latent_dim)
        decoded_flat = self.decoder(lat_flat)                        # (B*T, 16, 64, 64)
        decoded_obs = decoded_flat.view(B, T, C, H, W)              # (B, T, 16, 64, 64)

        reward_preds = self.reward_head(lat_flat).view(B, T)        # (B, T)
        continue_preds = self.continue_head(lat_flat).view(B, T)    # (B, T)

        return {
            "latents": latents_t,
            "prior_logits": prior_logits_t,
            "posterior_logits": posterior_logits_t,
            "decoded_obs": decoded_obs,
            "predicted_rewards": reward_preds,
            "predicted_continues": continue_preds,
        }

    # ------------------------------------------------------------------
    # Imagination rollout
    # ------------------------------------------------------------------

    def imagine(
        self,
        initial_h: torch.Tensor,
        initial_z: torch.Tensor,
        actor: HierarchicalActor,
        horizon: int,
    ) -> Dict[str, torch.Tensor]:
        """Roll out imagined trajectories using the actor policy.

        Starting from a given latent state, unrolls ``horizon`` steps using
        the prior (no real observations). The actor samples actions at each
        step. Gradients flow through the straight-through stochastic samples.

        Args:
            initial_h: (B, gru_hidden_dim) initial deterministic state.
            initial_z: (B, num_categories, num_classes) initial stochastic state.
            actor: HierarchicalActor used to select actions.
            horizon: Number of imagination steps.

        Returns:
            Dictionary containing:
                latents             : (B, horizon, latent_dim)
                action_types        : (B, horizon) int64 sampled actions
                action_xs           : (B, horizon) int64 click x (0 if non-click)
                action_ys           : (B, horizon) int64 click y (0 if non-click)
                log_probs           : (B, horizon) log probability of each action
                predicted_rewards   : (B, horizon) symlog reward predictions
                predicted_continues : (B, horizon) continue logits (pre-sigmoid)
        """
        h, z = initial_h, initial_z

        latents: List[torch.Tensor] = []
        action_types_list: List[torch.Tensor] = []
        action_xs_list: List[torch.Tensor] = []
        action_ys_list: List[torch.Tensor] = []
        log_probs_list: List[torch.Tensor] = []
        reward_preds_list: List[torch.Tensor] = []
        continue_preds_list: List[torch.Tensor] = []

        for _ in range(horizon):
            latent = self.rssm.get_latent(h, z)        # (B, latent_dim)
            latents.append(latent)

            # Actor selects action based on current latent
            action_type, x, y, log_prob, _ = actor.sample(latent)

            reward_pred = self.reward_head(latent).squeeze(-1)    # (B,)
            continue_pred = self.continue_head(latent).squeeze(-1)  # (B,)

            reward_preds_list.append(reward_pred)
            continue_preds_list.append(continue_pred)
            action_types_list.append(action_type)
            action_xs_list.append(x)
            action_ys_list.append(y)
            log_probs_list.append(log_prob)

            # Advance with prior (no observation)
            h, z, _ = self.rssm.imagine_step(h, z, action_type)

        return {
            "latents": torch.stack(latents, dim=1),
            "action_types": torch.stack(action_types_list, dim=1),
            "action_xs": torch.stack(action_xs_list, dim=1),
            "action_ys": torch.stack(action_ys_list, dim=1),
            "log_probs": torch.stack(log_probs_list, dim=1),
            "predicted_rewards": torch.stack(reward_preds_list, dim=1),
            "predicted_continues": torch.stack(continue_preds_list, dim=1),
        }
